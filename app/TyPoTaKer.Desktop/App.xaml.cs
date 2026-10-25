using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using System.Windows;
using TyPoTaKer.Desktop.Engine;
using TyPoTaKer.Desktop.Services;
using TyPoTaKer.Desktop.Views;

namespace TyPoTaKer.Desktop;

public partial class App : System.Windows.Application
{
    private TyPoInferenceEngine? _engine;
    private CaretOverlayWindow? _overlay;
    private KeyboardHook? _hook;
    private MouseHook? _mouseHook;
    private TrayManager? _tray;
    private System.Timers.Timer? _debounceTimer;
    private CancellationTokenSource? _inferenceCts;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        AppDomain.CurrentDomain.UnhandledException += (s, args) =>
        {
            try { File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash.log"), args.ExceptionObject?.ToString() ?? "Unknown error"); } catch { }
        };
        DispatcherUnhandledException += (s, args) =>
        {
            try { File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash.log"), args.Exception.ToString()); } catch { }
            args.Handled = true;
        };

        string root = FindWorkspaceRoot();
        string vocab = Path.Combine(root, @"data\tokenizer\vocab.json");
        string merges = Path.Combine(root, @"data\tokenizer\merges.txt");
        string lmStage2 = Path.Combine(root, @"export\lm_stage2_int8.onnx");
        string lm = File.Exists(lmStage2) ? lmStage2 : Path.Combine(root, @"export\lm_stage1_int8.onnx");

        // Prefer Stage 2 (94M params) GEC model and v4 maps if present
        string gecStage2 = Path.Combine(root, @"export\gec_stage2_int8.onnx");
        string gec = File.Exists(gecStage2) ? gecStage2 : Path.Combine(root, @"export\gec_stage1_int8.onnx");

        string mapsV4 = Path.Combine(root, @"data\processed\gec\v4\edit_maps.json");
        string mapsExport = Path.Combine(root, @"export\edit_maps.json");
        string mapsV3 = Path.Combine(root, @"data\processed\gec\v3\edit_maps.json");
        string maps = File.Exists(mapsV4) ? mapsV4 : (File.Exists(mapsExport) ? mapsExport : (File.Exists(mapsV3) ? mapsV3 : Path.Combine(root, @"data\processed\gec\v2\edit_maps.json")));

        try
        {
            File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "startup.log"),
                $"Time: {DateTime.Now}\nRoot: {root}\nLM: {lm} (Exists={File.Exists(lm)})\nGEC: {gec} (Exists={File.Exists(gec)})\nMaps: {maps} (Exists={File.Exists(maps)})\n");
        }
        catch { }

        try
        {
            _engine = new TyPoInferenceEngine(vocab, merges, lm, gec, maps);
            _overlay = new CaretOverlayWindow();
            _hook = new KeyboardHook();
            _mouseHook = new MouseHook();
            _tray = new TrayManager();

            // 60ms fast typing debouncer for snappy response
            _debounceTimer = new System.Timers.Timer(60) { AutoReset = false };
            _debounceTimer.Elapsed += OnDebounceElapsed;

            _hook.TypingActivity += (s, args) =>
            {
                _hook.IsDismissed = false;
                _debounceTimer.Stop();
                _debounceTimer.Start();
            };

            _hook.AcceptRequested += OnAcceptRequested;
            _hook.DismissRequested += (s, args) => _overlay.HideOverlay();
            _hook.SelectionDetected += (s, args) => _mouseHook.NotifyExternalSelection();
            _hook.SelectionCleared += (s, args) => _mouseHook.ClearSelection();

            _mouseHook.RightClickSelectionRequested += OnRightClickSelection;

            _overlay.SuggestionAccepted += OnAcceptRequested;
            _overlay.DismissClicked += (s, args) =>
            {
                if (_hook != null) _hook.IsDismissed = true;
                _overlay.HideOverlay();
            };

            _tray.EnabledToggled += (s, enabled) =>
            {
                if (_hook != null) _hook.IsEnabled = enabled;
                if (_mouseHook != null) _mouseHook.IsEnabled = enabled;
                if (!enabled) _overlay.HideOverlay();
            };

            _tray.ResetRequested += (s, args) =>
            {
                _hook.ClearBuffer();
                _mouseHook.ClearSelection();
                _overlay.HideOverlay();
                _tray.ShowNotification("Buffer Reset", "Typing context buffer cleared.");
            };

            _tray.ExitRequested += (s, args) =>
            {
                Shutdown();
            };

            _hook.Start();
            _mouseHook.Start();

            _tray.ShowNotification(
                "TyPoTaKer Active",
                "Ready for typing and right-click correction! Press Ctrl+Space to accept. Running in tray."
            );
        }
        catch (Exception ex)
        {
            System.Windows.MessageBox.Show(
                $"Failed to start TyPoTaKer: {ex.Message}",
                "TyPoTaKer Error",
                MessageBoxButton.OK,
                MessageBoxImage.Error
            );
            Shutdown();
        }
    }

    private void OnDebounceElapsed(object? sender, ElapsedEventArgs e)
    {
        if (_engine == null || _overlay == null || _hook == null || !_hook.IsEnabled || _hook.IsDismissed)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        _inferenceCts?.Cancel();
        var cts = new CancellationTokenSource();
        _inferenceCts = cts;
        var ct = cts.Token;

        Task.Run(() => ProcessSuggestionsAsync(ct), ct);
    }

    private void ProcessSuggestionsAsync(CancellationToken ct)
    {
        if (ct.IsCancellationRequested || _engine == null || _overlay == null || _hook == null)
            return;

        var caret = CaretTracker.GetCaretInfo();
        if (!caret.IsVisible || caret.IsPassword)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        string rawText = caret.SurroundingText ?? _hook.GetBuffer();
        if (string.IsNullOrWhiteSpace(rawText) || rawText.Trim().Length < 2)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        // Extract active sentence / line context
        string activeSentence = ExtractActiveSentence(rawText);
        if (string.IsNullOrWhiteSpace(activeSentence) || activeSentence.Length < 2)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        char lastChar = rawText[rawText.Length - 1];
        bool isWordBoundary = char.IsWhiteSpace(lastChar) || char.IsPunctuation(lastChar);

        // Only offer typing suggestions when a word, sentence, or line has been completed
        if (!isWordBoundary)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        if (ct.IsCancellationRequested) return;

        // 1. High-Priority: Check GEC grammar and typo corrections on the active sentence/phrase
        var gecResult = _engine.CorrectSentence(activeSentence, minEditProb: 0.65f, editMargin: 0.08f);
        if (gecResult.HasCorrection && gecResult.Edits.Count > 0)
        {
            if (ct.IsCancellationRequested) return;

            // Single word correction: target ONLY the mistyped word
            if (gecResult.Edits.Count == 1)
            {
                var edit = gecResult.Edits[0];
                string orig = edit.Original.Trim();
                string repl = edit.Replacement.Trim();

                if (!string.IsNullOrEmpty(orig) && !string.IsNullOrEmpty(repl) && !string.Equals(orig, repl, StringComparison.OrdinalIgnoreCase))
                {
                    int lastOrigIdx = activeSentence.LastIndexOf(orig, StringComparison.OrdinalIgnoreCase);
                    if (lastOrigIdx >= 0)
                    {
                        int backspaces = activeSentence.Length - lastOrigIdx;
                        string trailing = activeSentence.Substring(lastOrigIdx + orig.Length);
                        string insertion = repl + trailing;
                        string displayLabel = $"{orig} → {repl}";

                        _overlay.ShowSuggestion(
                            screenPos: caret.ScreenPosition,
                            heightOffset: caret.CaretHeight,
                            displayLabel: displayLabel,
                            insertionText: insertion,
                            badgeType: SuggestionBadgeType.Fix,
                            backspaces: backspaces,
                            isSelectionReplacement: false,
                            isAbovePosition: false
                        );
                        return;
                    }
                }
            }
            else
            {
                // Multi-error sentence correction
                string displayLabel = $"Fix: {gecResult.CorrectedText}";
                _overlay.ShowSuggestion(
                    screenPos: caret.ScreenPosition,
                    heightOffset: caret.CaretHeight,
                    displayLabel: displayLabel,
                    insertionText: gecResult.CorrectedText,
                    badgeType: SuggestionBadgeType.Fix,
                    backspaces: activeSentence.Length,
                    isSelectionReplacement: false,
                    isAbovePosition: false
                );
                return;
            }
        }

        if (ct.IsCancellationRequested) return;

        // 2. Second-Priority: Contextual autocomplete suggestions on the completed word boundary
        var suggestions = _engine.GetAutocomplete(activeSentence, topK: 1);
        if (suggestions.Count > 0 && suggestions[0].Probability >= 0.015f)
        {
            if (ct.IsCancellationRequested) return;

            var best = suggestions[0];
            string completion = best.Text.Trim();
            if (!string.IsNullOrWhiteSpace(completion))
            {
                // Autocomplete completions add a trailing space for smooth next-word typing
                string insertionText = completion + " ";

                _overlay.ShowSuggestion(
                    screenPos: caret.ScreenPosition,
                    heightOffset: caret.CaretHeight,
                    displayLabel: completion,
                    insertionText: insertionText,
                    badgeType: SuggestionBadgeType.Next,
                    backspaces: 0,
                    isSelectionReplacement: false,
                    isAbovePosition: false
                );
                return;
            }
        }

        Dispatcher.Invoke(() => _overlay?.HideOverlay());
    }

    private void OnRightClickSelection(object? sender, RightClickSelectionEventArgs e)
    {
        if (_engine == null || _overlay == null || _hook == null || !_hook.IsEnabled)
            return;

        Task.Run(() =>
        {
            // Tiny pause for the host app to update its selection/mouse state
            Thread.Sleep(30);

            string? selectedText = SelectionHelper.GetSelectedText(e.HasLikelySelection ? 100 : 40);
            if (string.IsNullOrWhiteSpace(selectedText) || selectedText.Trim().Length < 2)
            {
                // No text selected; do not interfere with standard right-click context menu
                return;
            }

            selectedText = selectedText.Trim();

            // Run GEC inference on the selected line/sentence
            var gecResult = _engine.CorrectSentence(selectedText, minEditProb: 0.65f, editMargin: 0.08f);

            if (gecResult.HasCorrection &&
                gecResult.Edits.Count > 0 &&
                !string.Equals(gecResult.CorrectedText.Trim(), selectedText, StringComparison.OrdinalIgnoreCase))
            {
                string displayLabel;
                if (gecResult.Edits.Count == 1)
                {
                    var edit = gecResult.Edits[0];
                    displayLabel = $"{edit.Original.Trim()} → {edit.Replacement.Trim()}";
                }
                else
                {
                    string corr = gecResult.CorrectedText.Trim();
                    displayLabel = corr.Length > 34 ? corr.Substring(0, 31) + "..." : corr;
                }

                _overlay.ShowSuggestion(
                    screenPos: e.ScreenPosition,
                    heightOffset: 0,
                    displayLabel: displayLabel,
                    insertionText: gecResult.CorrectedText,
                    badgeType: SuggestionBadgeType.Fix,
                    backspaces: 0,
                    isSelectionReplacement: true,
                    isAbovePosition: true
                );
            }
            else
            {
                // If doesn't have any changes needed, display '0 changes needed' and auto-dismiss
                _overlay.ShowCleanStatus(e.ScreenPosition, "0 changes needed");
            }
        });
    }

    private static string ExtractActiveSentence(string text)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;

        // Look back for sentence boundaries: newline or period/exclamation/question followed by space
        int lastBreak = -1;
        for (int i = text.Length - 1; i >= 0; i--)
        {
            char c = text[i];
            if (c == '\n' || c == '\r')
            {
                lastBreak = i;
                break;
            }
            if ((c == '.' || c == '!' || c == '?') && i < text.Length - 1 && text[i + 1] == ' ')
            {
                lastBreak = i + 1;
                break;
            }
        }

        string slice = (lastBreak >= 0 && lastBreak < text.Length)
            ? text.Substring(lastBreak + 1)
            : text;

        if (slice.Length > 160)
            slice = slice.Substring(slice.Length - 160);

        return slice;
    }

    private void OnAcceptRequested(object? sender, EventArgs e)
    {
        if (_overlay != null && _overlay.HasActiveSuggestion)
        {
            string insertion = _overlay.CurrentInsertion;
            int backspaces = _overlay.CurrentBackspaces;
            bool isSelection = _overlay.IsSelectionReplacement;
            _overlay.HideOverlay();

            if (isSelection)
            {
                // Text is currently selected in the active window.
                // In Windows, typing or sending Unicode characters into selected text replaces the selection directly.
                KeyboardHook.SendText(insertion, backspacesCount: 0);
                _hook?.ClearBuffer();
            }
            else
            {
                // Keystroke replacement with backspaces
                KeyboardHook.SendText(insertion, backspaces);
                _hook?.UpdateBufferOnAccept(backspaces, insertion);
            }

            // Immediately restart the debouncer on the newly updated buffer so the NEXT suggestion is accurate
            _debounceTimer?.Stop();
            _debounceTimer?.Start();
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _debounceTimer?.Dispose();
        _inferenceCts?.Cancel();
        _inferenceCts?.Dispose();
        _hook?.Dispose();
        _mouseHook?.Dispose();
        _tray?.Dispose();
        _engine?.Dispose();
        base.OnExit(e);
    }

    private static string FindWorkspaceRoot()
    {
        // 1. Search upwards from BaseDirectory for repository root containing pyproject.toml
        var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml")) &&
                Directory.Exists(Path.Combine(dir.FullName, "export")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        // 2. Current working directory
        string current = Directory.GetCurrentDirectory();
        if (File.Exists(Path.Combine(current, "pyproject.toml")) &&
            Directory.Exists(Path.Combine(current, "export")))
        {
            return current;
        }

        string desktopPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TyPoTaKer");
        if (File.Exists(Path.Combine(desktopPath, "pyproject.toml")))
            return desktopPath;

        return AppDomain.CurrentDomain.BaseDirectory;
    }
}
