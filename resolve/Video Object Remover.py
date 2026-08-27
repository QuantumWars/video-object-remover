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
import subprocess
import sys
import time

HANDOFF = os.path.expanduser(
    "~/Library/Application Support/VideoObjectRemover/resolve")
SESSION = os.path.join(HANDOFF, "session.json")
DONE = os.path.join(HANDOFF, "done.json")

APP = "/Applications/Video Object Remover.app"
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
    ("luma_matte", "Wire as a luminance matte",
     "Attach a greyscale matte to this clip's Fusion comp. Most flexible, "
     "least reliable — Resolve Free often ignores scripted comps."),
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
    return info


def ensure_track_above(timeline, track_index):
    """A video track above `track_index`, adding one only when needed."""
    count = int(timeline.GetTrackCount("video"))
    if count > track_index:
        return track_index + 1
    return track_index + 1 if timeline.AddTrack("video") else count


# --------------------------------------------------------------------------
# Handoff

def write_session(payload):
    os.makedirs(HANDOFF, exist_ok=True)
    for stale in (SESSION, DONE):
        if os.path.exists(stale):
            os.remove(stale)
    tmp = SESSION + ".tmp"
    with open(tmp, "w") as fh:
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
                with open(DONE) as fh:
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

def import_result(resolve_app, timeline, result, session):
    media_pool = resolve_app.GetProjectManager().GetCurrentProject().GetMediaPool()
    primary = result.get("primary")
    if not primary or not os.path.exists(primary):
        return False, "the app reported success but the file is missing"

    imported = media_pool.ImportMedia([primary])
    if not imported:
        return False, f"Resolve would not import {os.path.basename(primary)}"
    clip = imported[0]

    mode = session.get("return_mode", "plate_track")
    if mode == "media_pool":
        return True, f"Imported {os.path.basename(primary)} into the media pool."

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
        return False, "Imported, but Resolve refused to place it on the timeline."
    return True, f"Placed on V{track} at the same timecode."


def wire_luma_matte(timeline, clip, session):
    """Attach a matte to the clip's Fusion comp as an effect mask.

    This is the one route that depends on a scripted comp actually taking, and
    on Resolve Free it frequently does not: the nodes appear on the Fusion page
    and the render ignores them. Verified as far as the API allows, and honest
    about the rest.
    """
    item = timeline.GetCurrentVideoItem()
    if item is None:
        return False, "lost track of the clip under the playhead"
    try:
        comp = item.GetFusionCompByIndex(1) or item.AddFusionComp()
        if comp is None:
            return False, "could not open a Fusion comp for this clip"
        media_in = None
        for tool in comp.GetToolList(False, "MediaIn").values():
            media_in = tool
            break
        if media_in is None:
            return False, "the clip's comp has no MediaIn to attach a matte to"
        matte = comp.AddTool("MediaIn")
        if matte is None:
            return False, "Resolve would not add a MediaIn for the matte"
        media_in.EffectMask = matte.Output
        return True, ("Wired into the Fusion comp. Check the render — Resolve "
                      "Free often ignores comps built by a script.")
    except Exception as exc:                              # noqa: BLE001
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
        message("Video Object Remover",
                "Rendering the timeline span is not wired up yet — this build "
                "sends the original media. Re-run and choose 'Original media'.")
        return

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
