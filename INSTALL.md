<p align="center">
  <img src="assets/typoformer_logo.png" alt="TypoFormer Logo" width="480"/>
</p>

# TypoFormer — User Installation & Setup Guide

**TypoFormer** is an ultra-fast, 100% local, offline AI keystroke companion and grammar engine for Windows. Powered by our custom **Stage 2 (94M parameter) ONNX INT8 Language Model & Edit-Tagging GEC**, TypoFormer provides Grammarly-grade real-time suggestions and zero-lag autocomplete anywhere in Windows in under **15 milliseconds** per keystroke without sending a single byte to the cloud.

---

## System Requirements

| Requirement | Specification |
| :--- | :--- |
| **Operating System** | Windows 10 (Build 1809+) or Windows 11 (64-bit) |
| **Processor** | Any x64 Intel/AMD CPU (Zero GPU required for inference) |
| **Memory** | 250 MB RAM |
| **Disk Space** | ~450 MB (Includes quantized Stage 2 neural models & tokenizers) |
| **Runtime** | [.NET 8 Desktop Runtime](https://dotnet.microsoft.com/download/dotnet/8.0) *(installed automatically if needed)* |

---

## Installation Methods

### Method 1: 1-Click Auto Installer (Recommended for End Users)

The easiest way to install and integrate TypoFormer with your Windows desktop and Start Menu:

1. Open the [`installer/`](installer) folder.
2. Double-click **`Install.bat`** (or right-click and select **Run as Administrator** if your organization requires it).
3. The installer will:
   - Copy all application binaries, neural models, tokenizers, and icons to `%LOCALAPPDATA%\Programs\TypoFormer`.
   - Create a **Desktop Shortcut** (`TypoFormer.lnk`) with the official TypoFormer icon.
   - Create a **Start Menu Shortcut** in Windows Programs.
   - Configure optional Windows logon auto-start.
   - Automatically launch TypoFormer in your Windows System Tray.

> [!TIP]
> No administrator rights are required for normal users since it installs cleanly to your local user profile.

---

### Method 2: Portable / Standalone Mode (Zero Installation)

If you prefer to run TypoFormer portably (e.g. from a USB drive or without modifying your system):

1. Navigate to:
   ```
   app\bin\publish\
   ```
2. Double-click **`TyPoTaKer.Desktop.exe`**.
3. TypoFormer will start immediately in your System Tray and begin monitoring keystrokes.

---

## How to Use TypoFormer

### 1. In-Line Typing Autocomplete
* Open **Notepad**, **Microsoft Word**, **Google Docs**, **VS Code**, or any text field.
* Type a sentence or thought naturally (e.g., `here is the `).
* A sleek floating glassmorphic pill will appear smoothly right below your blinking text cursor with a prediction badge:
  * <span style="background-color: #6366F1; color: white; padding: 2px 6px; border-radius: 4px;">NEXT</span> Contextual next-word autocomplete
  * <span style="background-color: #10B981; color: white; padding: 2px 6px; border-radius: 4px;">FIX</span> Real-time typo or grammar correction
* **Accept the suggestion**:
  * Press <kbd>Tab</kbd> or <kbd>Ctrl</kbd> + <kbd>Space</kbd> (or click the suggestion pill directly).
  * The suggestion will be inserted instantly with smart spacing, and the next prediction will adapt immediately!

### 2. Right-Click Instant Correction
* Highlight any paragraph, sentence, or word in any application.
* **Right-click** on the selected text.
* TypoFormer scans the text through the GEC Neural Engine and presents the corrected version floating above your cursor.
* Click the suggestion or press <kbd>Ctrl</kbd> + <kbd>Space</kbd> to replace the selection with the clean text.

### 3. Modern Control Dashboard
Double-click the **TypoFormer icon** in the bottom-right Windows System Tray (or right-click the tray icon and choose **Open Dashboard...**):
* **Live Interactive Playground**: Type and test model suggestions without switching apps.
* **Model Diagnostic Badges**: Verify Stage 2 INT8 LM and GEC loading status.
* **Preferences**:
  * Toggle <kbd>Tab</kbd> key suggestion acceptance.
  * Toggle Start with Windows on/off.
  * One-click **Reset Context Buffer** button.

---

## How to Uninstall

If you ever wish to remove TypoFormer completely:

1. Open the [`installer/`](installer) folder.
2. Double-click **`Uninstall.bat`**.
3. All shortcuts, auto-start entries, and program files will be completely deleted in seconds.

---

## Diagnostics & Troubleshooting

* **Check Startup Logs**:
  Look at `startup.log` inside `%LOCALAPPDATA%\Programs\TypoFormer\startup.log` (or `app\bin\publish\startup.log`).
  It displays:
  ```
  LM: ...\export\lm_stage2_int8.onnx (Exists=True)
  GEC: ...\export\gec_stage2_int8.onnx (Exists=True)
  Maps: ...\edit_maps.json (Exists=True)
  ```
* **Antivirus False Positives**:
  Because TypoFormer uses Windows low-level keyboard hooks (`SetWindowsHookEx`) to detect caret typing boundaries across arbitrary applications, some strict security software may flag new unsigned binaries. If prompted, add an exclusion for `%LOCALAPPDATA%\Programs\TypoFormer\TyPoTaKer.Desktop.exe`.
