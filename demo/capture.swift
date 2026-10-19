// Records one window, and nothing else on the screen, as PNG frames: ScreenCaptureKit with a filter for that window
// alone (covering windows and the rest of the desktop never reach it). Frames arrive when the window changes, so each
// file is named after its time in milliseconds; record.sh turns them into the GIF.
//
//     swift demo/capture.swift <pid of the window's app> <seconds> <folder>

import AppKit
import CoreImage
import Foundation
import ScreenCaptureKit

let args = CommandLine.arguments
guard args.count == 4, let pid = Int32(args[1]), let seconds = Double(args[2]) else {
    FileHandle.standardError.write("usage: capture.swift <pid> <seconds> <folder>\n".data(using: .utf8)!)
    exit(2)
}
let folder = URL(fileURLWithPath: args[3])
try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

final class Recorder: NSObject, SCStreamOutput {
    let started = Date()
    let context = CIContext()
    var frames = 0

    func stream(_ stream: SCStream, didOutputSampleBuffer buffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, let pixels = CMSampleBufferGetImageBuffer(buffer) else { return }
        // Only complete frames carry new pixels.
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(buffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let raw = attachments.first?[.status] as? Int, SCFrameStatus(rawValue: raw) != .complete {
            return
        }
        let image = CIImage(cvPixelBuffer: pixels)
        let ms = Int(Date().timeIntervalSince(started) * 1000)
        let url = folder.appendingPathComponent(String(format: "%07d.png", ms))
        try? context.writePNGRepresentation(of: image, to: url, format: .RGBA8,
                                             colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        frames += 1
    }
}

let recorder = Recorder()
Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
        let windows = content.windows.filter { $0.owningApplication?.processID == pid && $0.frame.height > 200 }
        guard let window = windows.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }) else {
            FileHandle.standardError.write("no window for process \(pid)\n".data(using: .utf8)!)
            exit(1)
        }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        let scale = Int(NSScreen.main?.backingScaleFactor ?? 2)
        config.width = Int(window.frame.width) * scale
        config.height = Int(window.frame.height) * scale
        config.minimumFrameInterval = CMTime(value: 1, timescale: 20)
        config.showsCursor = false
        config.capturesAudio = false
        let stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream.addStreamOutput(recorder, type: .screen, sampleHandlerQueue: DispatchQueue(label: "frames"))
        try await stream.startCapture()
        print("recording", window.title ?? "", Int(window.frame.width), "x", Int(window.frame.height))
        try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        try? await stream.stopCapture()   // already stopped if the window closed first
        print("\(recorder.frames) frames")
        exit(0)
    } catch {
        FileHandle.standardError.write("capture failed: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
RunLoop.main.run()
