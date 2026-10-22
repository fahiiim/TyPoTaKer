# TyPoTaKer Desktop Assistant (Native Windows .NET 8)

The always-on, ultra-low-latency Windows desktop companion for TyPoTaKer.

## Features

- **System Tray Daemon**: Runs silently in the Windows system tray with zero background window clutter.
- **Glassmorphic Floating Caret Pill**: Smooth, non-activating hardware-accelerated overlay positioned dynamically just below the caret in any active application (Notepad, VS Code, Word, Chrome, etc.).
- **Pure Offline ONNX Runtime**: Uses `Microsoft.ML.OnnxRuntime` with INT8 dynamic quantization for sub-22 ms keystroke response on standard CPU.
- **Privacy & Security**: Built-in Windows UI Automation password detection (`AutomationElement.IsPasswordProperty`) suppresses buffering and suggestions on all password and credential fields.
- **Global Hotkey Acceptance**: Press `Ctrl + Space` to instantly inject the completion or replace the mistyped word via Win32 `SendInput`.
- **Zero Interception of Tab**: Preserves standard code and text navigation shortcuts.

## Architecture

```text
app/TyPoTaKer.Desktop/
├── Engine/
│   ├── TyPoTokenizer.cs        # Bit-for-bit ByteLevel GPT-2 BPE tokenizer + char-CNN generator
│   └── TyPoInferenceEngine.cs   # ONNX Runtime session manager for LM & GEC INT8 heads
├── Services/
│   ├── Win32Native.cs          # Centralized P/Invoke (hooks, GUI thread info, SendInput)
│   ├── CaretTracker.cs         # UI Automation TextPattern + GetGUIThreadInfo fallback
│   ├── KeyboardHook.cs         # Low-level WH_KEYBOARD_LL hook & text injection
│   └── TrayManager.cs          # Windows system tray icon & status context menu
├── Views/
│   ├── CaretOverlayWindow.xaml # Glassmorphic floating pill UI (WS_EX_NOACTIVATE)
│   └── CaretOverlayWindow.xaml.cs
├── App.xaml                    # Background WPF application definition (ShutdownMode=OnExplicitShutdown)
└── App.xaml.cs                 # Real-time coordinator connecting hook, engine, and overlay
```

## Build & Run

### Run from Source
```powershell
dotnet run --project app/TyPoTaKer.Desktop/TyPoTaKer.Desktop.csproj
```

### Build Release Binary
```powershell
dotnet publish app/TyPoTaKer.Desktop/TyPoTaKer.Desktop.csproj -c Release -r win-x64 --self-contained false -o app/bin/publish
```

The published release binary is located at `app/bin/publish/TyPoTaKer.Desktop.exe` (~150 KB, ~12 MB total with native ONNX Runtime binaries).
