"""Send the clip under the playhead to Video Object Remover, and bring the
result back.

Install into Resolve's Utility scripts folder (the app can do this for you):

    ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/
        Fusion/Scripts/Utility/Video Object Remover.py

Then: Workspace > Scripts > Utility > Video Object Remover.

Two things shape the design.

Resolve Free will not render a Fusion comp that was built through the scripting
API — nodes can be created, wired and keyframed, and the comp simply never
takes. So the default routes avoid Fusion entirely: a removal comes back as a
finished plate, and a matte comes back as ProRes 4444 whose alpha Resolve reads
natively on import. Both are just clips on a track. The luminance-matte route is
offered because it is genuinely more flexible for grading, but it is the one
path that depends on scripted comps working, and it says so.

Second, the script does as little as possible. It identifies the clip, hands
over a JSON file, waits, and imports what comes back. Everything the user
actually interacts with happens in the app, which is better at it.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HANDOFF = os.path.expanduser(
    "~/Library/Application Support/VideoObjectRemover/resolve")
SESSION = os.path.join(HANDOFF, "session.json")
DONE = os.path.join(HANDOFF, "done.json")

APP = "/Applications/Video Object Remover.app"
MATTE_NODE = "VOR_Matte"
RENDER_DIR = os.path.join(HANDOFF, "render")
#: ProRes 422 HQ: visually lossless for this, and available on Resolve Free.
RENDER_FORMAT, RENDER_CODEC = "mov", "ProRes422HQ"
#: throwaway timeline used to isolate the clip while rendering
SCRATCH_TIMELINE = "VOR scratch (safe to delete)"
#: media-pool bin and on-disk folder the results land in
RESULT_BIN = "Video Object Remover"
POLL_SECONDS = 1.0
TIMEOUT_SECONDS = 4 * 60 * 60          # a 4K removal is genuinely hours

SOURCE_MODES = [
    ("file", "Original media",
     "Process the clip's source file. Instant, full quality, ignores grades."),
    ("render", "Timeline render",
     "Render this clip's span first, so grades and transforms are included."),
]

RETURN_MODES = [
    ("plate_track", "New track above",
     "Put the result on a video track above this clip, at the same timecode."),
    ("media_pool", "Media pool only",
     "Import it and leave the timeline alone."),
    ("luma_matte", "Add a matte node to the comp",
     "For driving a grade or effect. To cut the object out, use 'New track "
     "above' with the ProRes 4444 output instead."),
]


# --------------------------------------------------------------------------
# Resolve handles

def get_resolve():
    try:
        return resolve                                   # noqa: F821
    except NameError:
        pass
    try:
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")
    except ImportError:
        sys.path.append("/Library/Application Support/Blackmagic Design/"
                        "DaVinci Resolve/Developer/Scripting/Modules")
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")


def get_fusion():
    try:
        return fusion                                    # noqa: F821
    except NameError:
        return get_resolve().Fusion()


# --------------------------------------------------------------------------
# Dialogs
#
# Resolve's own UI toolkit, so the window looks native and sits with the app
# rather than behind it.

def message(title, text):
    fu = get_fusion()
    ui, disp = fu.UIManager, __import__("bmd").UIDispatcher(fu.UIManager)
    win = disp.AddWindow(
        {"WindowTitle": title, "ID": "vorMsg",
         "Geometry": [500, 300, 460, 170]},
        [ui.VGroup({"Spacing": 12}, [
            ui.Label({"Text": text, "WordWrap": True, "Alignment": {"AlignTop": True}}),
            ui.HGroup({"Weight": 0}, [
                ui.HGap(0, 1.0),
                ui.Button({"ID": "ok", "Text": "OK", "Weight": 0}),
            ]),
        ])])
    win.On.vorMsg.Close = lambda ev: disp.ExitLoop()
    win.On.ok.Clicked = lambda ev: disp.ExitLoop()
    win.Show()
    disp.RunLoop()
    win.Hide()


def ask_options(clip_name):
    """Pick how the clip goes out and how the result comes back."""
    fu = get_fusion()
    ui, disp = fu.UIManager, __import__("bmd").UIDispatcher(fu.UIManager)
    chosen = {}

    def radio_group(prefix, options, default=0):
        rows = []
        for i, (key, label, blurb) in enumerate(options):
            rows.append(ui.VGroup({"Spacing": 1, "Weight": 0}, [
                ui.CheckBox({"ID": f"{prefix}{i}", "Text": label,
                             "Checked": i == default, "Weight": 0}),
                ui.Label({"Text": "     " + blurb, "WordWrap": True,
                          "Weight": 0, "StyleSheet": "color: #999;"}),
            ]))
        return rows

    win = disp.AddWindow(
        {"WindowTitle": "Video Object Remover", "ID": "vorOpts",
         "Geometry": [400, 200, 520, 470]},
        [ui.VGroup({"Spacing": 10}, [
            ui.Label({"Text": f"<b>{clip_name}</b>", "Weight": 0}),
            ui.Label({"Text": "Send to app", "Weight": 0,
                      "StyleSheet": "font-weight: bold; margin-top: 6px;"}),
        ] + radio_group("src", SOURCE_MODES) + [
            ui.Label({"Text": "Bring the result back as", "Weight": 0,
                      "StyleSheet": "font-weight: bold; margin-top: 6px;"}),
        ] + radio_group("ret", RETURN_MODES) + [
            ui.VGap(0, 1.0),
            ui.HGroup({"Weight": 0}, [
                ui.HGap(0, 1.0),
                ui.Button({"ID": "cancel", "Text": "Cancel", "Weight": 0}),
                ui.Button({"ID": "go", "Text": "Continue", "Weight": 0}),
            ]),
        ])])

    items = win.GetItems()

    # Checkboxes behave as radios: selecting one clears its siblings. Resolve's
    # UIManager has no radio group, and two "sources" selected at once is not a
    # state the rest of the script should have to think about.
    def exclusive(prefix, count):
        def handler(idx):
            def on_click(ev):
                for j in range(count):
                    items[f"{prefix}{j}"].Checked = (j == idx)
            return on_click
        for i in range(count):
            setattr(win.On, f"{prefix}{i}", None)
            win.On[f"{prefix}{i}"].Clicked = handler(i)

    exclusive("src", len(SOURCE_MODES))
    exclusive("ret", len(RETURN_MODES))

    def finish(ev):
        for i, (key, _, _) in enumerate(SOURCE_MODES):
            if items[f"src{i}"].Checked:
                chosen["source_mode"] = key
        for i, (key, _, _) in enumerate(RETURN_MODES):
            if items[f"ret{i}"].Checked:
                chosen["return_mode"] = key
        disp.ExitLoop()

    win.On.vorOpts.Close = lambda ev: disp.ExitLoop()
    win.On.cancel.Clicked = lambda ev: disp.ExitLoop()
    win.On.go.Clicked = finish
    win.Show()
    disp.RunLoop()
    win.Hide()
    return chosen if chosen.get("source_mode") else None


def progress_window(clip_name):
    """A window that stays up while the app works, so Resolve does not look
    frozen and the user has a way out."""
    fu = get_fusion()
    ui, disp = fu.UIManager, __import__("bmd").UIDispatcher(fu.UIManager)
    state = {"cancelled": False}
    win = disp.AddWindow(
        {"WindowTitle": "Video Object Remover", "ID": "vorWait",
         "Geometry": [400, 200, 460, 180]},
        [ui.VGroup({"Spacing": 10}, [
            ui.Label({"Text": f"<b>{clip_name}</b>", "Weight": 0}),
            ui.Label({"ID": "status", "Text": "Working in Video Object Remover…",
                      "WordWrap": True}),
            ui.Label({"Text": "You can keep using Resolve. Closing this window "
                              "stops waiting; it does not stop the render.",
                      "WordWrap": True, "Weight": 0,
                      "StyleSheet": "color: #999; font-size: 11px;"}),
            ui.HGroup({"Weight": 0}, [
                ui.HGap(0, 1.0),
                ui.Button({"ID": "stop", "Text": "Stop waiting", "Weight": 0}),
            ]),
        ])])

    def cancel(ev):
        state["cancelled"] = True
        disp.ExitLoop()
    win.On.vorWait.Close = cancel
    win.On.stop.Clicked = cancel
    return win, disp, state


# --------------------------------------------------------------------------
# Timeline

def clip_under_playhead(timeline):
    """The video item at the playhead, plus where it sits.

    GetCurrentVideoItem() follows the *current* video track, which is what the
    user means by "this clip" — but it does not say which track that was, and
    the result has to land above it.
    """
    item = timeline.GetCurrentVideoItem()
    if item is None:
        return None, None
    for track in range(1, int(timeline.GetTrackCount("video")) + 1):
        for candidate in timeline.GetItemListInTrack("video", track) or []:
            if candidate == item or (
                    candidate.GetName() == item.GetName()
                    and candidate.GetStart() == item.GetStart()):
                return item, track
    return item, 1


def describe(resolve_app, timeline, item, track_index):
    pool_item = item.GetMediaPoolItem()
    fps = float(timeline.GetSetting("timelineFrameRate")
                or resolve_app.GetProjectManager().GetCurrentProject()
                .GetSetting("timelineFrameRate") or 24)
    info = {
        "clip_name": item.GetName(),
        "track_index": int(track_index),
        "record_frame": int(item.GetStart()),
        "record_end": int(item.GetEnd()),
        "duration": int(item.GetDuration()),
        "left_offset": int(item.GetLeftOffset()),
        "fps": fps,
        "timeline": timeline.GetName(),
    }
    if pool_item is not None:
        info["file_path"] = pool_item.GetClipProperty("File Path")
        info["source_start"] = int(item.GetLeftOffset())
        info["source_end"] = int(item.GetLeftOffset()) + int(item.GetDuration())

    # Write results into Resolve's own media folder rather than beside whatever
    # the source happened to be. It shows up in Media Storage, it is where an
    # editor looks for renders, and it survives the source being on a volume
    # that is not always mounted.
    project = resolve_app.GetProjectManager().GetCurrentProject()
    media_root = project.GetSetting("projectMediaLocation") or ""
    if media_root and os.path.isdir(media_root):
        info["output_dir"] = os.path.join(media_root, RESULT_BIN, project.GetName())
    else:
        info["output_dir"] = os.path.join(
            os.path.dirname(info.get("file_path") or ""), RESULT_BIN)
    info["project"] = project.GetName()
    return info


def ensure_track_above(timeline, track_index):
    """A video track above `track_index`, adding one only when needed."""
    count = int(timeline.GetTrackCount("video"))
    if count > track_index:
        return track_index + 1
    return track_index + 1 if timeline.AddTrack("video") else count


# --------------------------------------------------------------------------
# Rendering the timeline span

def render_span(project, timeline, session, on_status=None):
    """Render just this clip's span, as the timeline sees it.

    Rendering rather than reading the source file is the point of this mode:
    grades, resizes and effects are baked in, so a matte lines up with the
    picture the editor is actually looking at.

    Anything on a higher track would composite over the clip, so it has to go.
    `SetTrackEnable` looks like the way to do that and is not — measured on
    21.0.4.5 it returns True and leaves the track enabled, and the render came
    back with the upper clip still in it. So the work happens on a *duplicate*
    timeline, where clips above the target can simply be deleted. The user's
    own timeline is never modified; the copy is thrown away either way.
    """
    say = on_status or (lambda _t: None)
    if os.path.isdir(RENDER_DIR):
        shutil.rmtree(RENDER_DIR, ignore_errors=True)
    os.makedirs(RENDER_DIR, exist_ok=True)

    keep = int(session.get("track_index", 1))
    media_pool = project.GetMediaPool()
    scratch = None
    job_id = None
    try:
        say("Preparing a copy of the timeline…")
        scratch = timeline.DuplicateTimeline(SCRATCH_TIMELINE)
        if scratch is None:
            return None, ("Resolve would not duplicate the timeline, which is "
                          "how this mode isolates the clip. Use 'Original "
                          "media' instead.")
        project.SetCurrentTimeline(scratch)

        for track in range(int(scratch.GetTrackCount("video")), keep, -1):
            items = scratch.GetItemListInTrack("video", track) or []
            if items:
                scratch.DeleteClips(items)

        if not project.SetCurrentRenderFormatAndCodec(RENDER_FORMAT, RENDER_CODEC):
            return None, "Resolve would not select a ProRes render preset"

        ok = project.SetRenderSettings({
            "SelectAllFrames": False,
            "MarkIn": int(session["record_frame"]),
            # Inclusive here, unlike AppendToTimeline's endFrame.
            "MarkOut": int(session["record_end"]) - 1,
            "TargetDir": RENDER_DIR,
            "CustomName": "vor_span",
            "UniqueFilenameStyle": 0,
            "ExportVideo": True,
            "ExportAudio": False,
        })
        if not ok:
            return None, "Resolve rejected the render settings"

        job_id = project.AddRenderJob()
        if not job_id:
            return None, "Resolve would not queue the render"
        project.StartRendering(job_id)
        while project.IsRenderingInProgress():
            status = project.GetRenderJobStatus(job_id) or {}
            say(f"Rendering the timeline span… "
                f"{status.get('CompletionPercentage', 0)}%")
            time.sleep(POLL_SECONDS)

        status = project.GetRenderJobStatus(job_id) or {}
        if status.get("JobStatus") != "Complete":
            return None, f"the render did not finish ({status.get('JobStatus')})"

        rendered = [os.path.join(RENDER_DIR, f)
                    for f in sorted(os.listdir(RENDER_DIR))
                    if f.lower().endswith(".mov")]
        if not rendered:
            return None, "the render reported success but produced no file"
        return rendered[0], None
    finally:
        try:
            project.SetCurrentTimeline(timeline)
        except Exception:                                     # noqa: BLE001
            pass
        if scratch is not None:
            try:
                media_pool.DeleteTimelines([scratch])
            except Exception:                                 # noqa: BLE001
                pass
        if job_id:
            try:
                project.DeleteRenderJob(job_id)
            except Exception:                                 # noqa: BLE001
                pass


# --------------------------------------------------------------------------
# Handoff

def write_session(payload):
    os.makedirs(HANDOFF, exist_ok=True)
    for stale in (SESSION, DONE):
        if os.path.exists(stale):
            os.remove(stale)
    tmp = SESSION + ".tmp"
    # Explicit UTF-8: Resolve's embedded Python runs under an ASCII locale, so
    # the default encoding cannot represent a clip named "Café.mov" and fails
    # with a UnicodeDecodeError that says nothing about clip names.
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, SESSION)            # the app must never see a partial file


def launch_app():
    if os.path.isdir(APP):
        subprocess.run(["open", "-a", APP], check=False)
        return True
    return False


def wait_for_done(win, disp, state):
    """Poll for the app's answer while keeping the window responsive."""
    deadline = time.time() + TIMEOUT_SECONDS
    win.Show()
    while time.time() < deadline and not state["cancelled"]:
        if os.path.exists(DONE):
            try:
                with open(DONE, encoding="utf-8") as fh:
                    result = json.load(fh)
                win.Hide()
                return result
            except (OSError, ValueError):
                pass                    # still being written; try again
        # RunLoop would block until the window closes, so step the dispatcher
        # instead and keep polling.
        disp.StepLoop(POLL_SECONDS * 1000)
    win.Hide()
    return None


# --------------------------------------------------------------------------
# Bringing it back

def result_bin(media_pool):
    """A bin for results, so they are easy to find and drag rather than loose
    in whatever folder happened to be selected."""
    root = media_pool.GetRootFolder()
    for folder in (root.GetSubFolderList() or []):
        if folder.GetName() == RESULT_BIN:
            return folder
    return media_pool.AddSubFolder(root, RESULT_BIN) or root


def import_result(resolve_app, timeline, result, session):
    media_pool = resolve_app.GetProjectManager().GetCurrentProject().GetMediaPool()
    primary = result.get("primary")
    if not primary or not os.path.exists(primary):
        return False, "the app reported success but the file is missing"

    # Import everything the run produced, not just the primary — a matte run
    # can write three files and the other two are no use sitting on disk.
    paths = [p for p in (result.get("outputs") or {}).values()
             if p and os.path.isfile(p)]
    if primary not in paths:
        paths.insert(0, primary)

    previous = media_pool.GetCurrentFolder()
    imported = []
    try:
        media_pool.SetCurrentFolder(result_bin(media_pool))
        imported = media_pool.ImportMedia(paths) or []
    finally:
        if previous:
            media_pool.SetCurrentFolder(previous)
    if not imported:
        return False, f"Resolve would not import {os.path.basename(primary)}"
    # ImportMedia returns them in the order given, so the primary is first.
    clip = imported[0]

    mode = session.get("return_mode", "plate_track")
    extra = f" ({len(imported)} files)" if len(imported) > 1 else ""
    if mode == "media_pool":
        return True, (f"In the '{RESULT_BIN}' bin{extra} — drag it to the "
                      f"timeline when you want it.")

    if mode == "luma_matte":
        ok, detail = wire_luma_matte(timeline, clip, session)
        if ok:
            return True, detail
        return False, detail + " The clip is in the media pool."

    track = ensure_track_above(timeline, session.get("track_index", 1))
    frames = int(clip.GetClipProperty("Frames") or session.get("duration", 0))
    appended = media_pool.AppendToTimeline([{
        "mediaPoolItem": clip,
        "startFrame": 0,
        # endFrame is EXCLUSIVE. The intuitive `frames - 1` lands the clip one
        # frame short, silently, and a one-frame gap at the tail of a matte is
        # exactly the sort of thing nobody notices until delivery.
        "endFrame": max(1, frames),
        "trackIndex": track,
        "recordFrame": session.get("record_frame", 0),
    }])
    if not appended:
        return False, (f"In the '{RESULT_BIN}' bin, but Resolve refused to "
                       f"place it on the timeline — drag it across.")
    return True, (f"Placed on V{track} at the same timecode, and in the "
                  f"'{RESULT_BIN}' bin{extra}.")


def wire_luma_matte(timeline, clip, session):
    """Import the matte and wire it into the clip's Fusion comp.

    Measured on Resolve 21.0.4.5 (Free), against a rendered frame each time:

      * a comp built by a script *does* render — an inserted Blur dropped
        frame sharpness from 77.7 to 29.2, so the older belief that scripted
        comps are ignored does not hold on this version
      * a MediaIn bound through SetInput("MediaID", ...) resolves and renders
        the matte
      * EffectMask genuinely limits an effect: the same Blur behind this matte
        was suppressed to 0.18 against an unblurred frame, versus 5.91 for a
        full-frame blur

    What it does *not* do is cut the picture. An EffectMask restricts the
    effect a node applies, and a MediaIn applies none, so hanging a matte on it
    changes nothing visible — measured at 0.02 mean difference, with
    MultiplyByMask set. The node is still wired and ready to drive whatever the
    colourist puts after it, which is the useful half; for an actual cut-out
    the ProRes 4444 output carries a real alpha channel and needs none of this.
    """
    item = timeline.GetCurrentVideoItem()
    if item is None:
        return False, "lost track of the clip under the playhead"
    try:
        comp = item.GetFusionCompByIndex(1) or item.AddFusionComp()
        if comp is None:
            return False, "could not open a Fusion comp for this clip"

        media_in = None
        for tool in (comp.GetToolList(False, "MediaIn") or {}).values():
            if tool.Name != MATTE_NODE:
                media_in = tool
                break
        if media_in is None:
            return False, "the clip's comp has no MediaIn to attach a matte to"

        # Replace our own previous matte rather than stacking a new one on
        # every run.
        for tool in list((comp.GetToolList(False) or {}).values()):
            if tool.Name == MATTE_NODE:
                try:
                    tool.Delete()
                except Exception:                             # noqa: BLE001
                    pass

        # A Fusion MediaIn inside Resolve binds to a media-pool item by its
        # MediaID, not to a path — a Loader pointed at the file is a different
        # node and behaves differently inside a clip comp.
        media_id = None
        try:
            media_id = clip.GetMediaId()
        except Exception:                                     # noqa: BLE001
            pass
        if not media_id:
            return False, "could not resolve the imported matte's media id"

        node = comp.AddTool("MediaIn", -2, 0)
        if node is None:
            return False, "Resolve would not add a MediaIn for the matte"
        node.SetInput("MediaID", media_id)
        try:
            node.SetAttrs({"TOOLS_Name": MATTE_NODE})
        except Exception:                                     # noqa: BLE001
            pass

        media_in.EffectMask = node.Output
        media_in.SetInput("MaskChannel", 5)          # 5 = luminance

        note = (f"Matte added to the Fusion comp as '{MATTE_NODE}'. Hang an "
                f"effect off it — it does not cut the picture by itself.")
        # A matte made from the original media is at the *source* resolution,
        # while the comp works at timeline resolution. They only line up when
        # the two match, which is what "Timeline render" guarantees.
        if session.get("source_mode") == "file":
            note += (" It was made from the original media, so if this clip is "
                     "scaled or letterboxed on the timeline the matte will not "
                     "line up — re-run with 'Timeline render' if so.")
        return True, note
    except Exception as exc:                                  # noqa: BLE001
        return False, f"could not wire the matte: {exc}"


# --------------------------------------------------------------------------

def main():
    resolve_app = get_resolve()
    if resolve_app is None:
        print("Could not reach Resolve.")
        return

    project = resolve_app.GetProjectManager().GetCurrentProject()
    if project is None:
        message("Video Object Remover", "Open a project first.")
        return
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        message("Video Object Remover", "Open a timeline first.")
        return

    item, track_index = clip_under_playhead(timeline)
    if item is None:
        message("Video Object Remover",
                "Put the playhead over a clip on a video track, then run this "
                "again.")
        return

    session = describe(resolve_app, timeline, item, track_index)
    if not session.get("file_path"):
        message("Video Object Remover",
                f"'{session['clip_name']}' has no media file behind it "
                f"(a compound clip, title or generator). Only clips backed by "
                f"a file can be sent.")
        return

    options = ask_options(session["clip_name"])
    if options is None:
        return
    session.update(options)
    session["version"] = 1
    session["created"] = time.time()

    if session["source_mode"] == "render":
        win, disp, state = progress_window(session["clip_name"])
        win.Show()

        def status(text):
            try:
                win.GetItems()["status"].Text = text
                disp.StepLoop(1)
            except Exception:                                 # noqa: BLE001
                pass

        rendered, error = render_span(project, timeline, session, status)
        win.Hide()
        if rendered is None:
            message("Video Object Remover", error or "The render failed.")
            return
        # The app now works on the rendered span, which starts at frame 0 —
        # but the result still has to land at the clip's own timecode.
        session["file_path"] = rendered
        session["rendered_from"] = session.get("file_path")
        session["source_start"] = 0
        session["source_end"] = int(session["duration"])

    write_session(session)
    if not launch_app():
        message("Video Object Remover",
                f"Could not find {APP}. Install the app, or start it yourself "
                f"— it will pick up this clip either way.")
        return

    win, disp, state = progress_window(session["clip_name"])
    result = wait_for_done(win, disp, state)

    if result is None:
        return                                   # cancelled or timed out
    if result.get("status") != "done":
        message("Video Object Remover",
                result.get("error") or "The app did not finish.")
        return

    ok, detail = import_result(resolve_app, timeline, result, session)
    message("Video Object Remover", detail)


main()
