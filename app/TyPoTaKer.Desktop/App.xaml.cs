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
    private DashboardWindow? _dashboard;
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
            _dashboard = new DashboardWindow();
            MainWindow = _dashboard;
            _dashboard.SetEngine(_engine);
            _overlay = new CaretOverlayWindow();
            _hook = new KeyboardHook();
            _mouseHook = new MouseHook();
            _tray = new TrayManager();

            // Intercept Tab acceptance when an active suggestion is visible and Tab accept is enabled
            _hook.HasActiveSuggestionPredicate = () => _overlay?.HasActiveSuggestion == true && _overlay.AllowTabAccept;

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

            _tray.DashboardRequested += (s, args) =>
            {
                Dispatcher.Invoke(() =>
                {
                    if (_dashboard != null)
                    {
                        _dashboard.Show();
                        _dashboard.WindowState = WindowState.Normal;
                        _dashboard.Activate();
                    }
                });
            };

            _dashboard.TabAcceptToggled += (s, allowed) =>
            {
                if (_overlay != null) _overlay.AllowTabAccept = allowed;
            };

            _dashboard.ResetBufferRequested += (s, args) =>
            {
                _hook?.ClearBuffer();
                _overlay?.HideOverlay();
                _tray.ShowNotification("Buffer Reset", "Context buffer wiped from Dashboard.");
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
                "TypoFormer Active",
                "Ready for typing and right-click correction! Press Tab or Ctrl+Space to accept. Running in tray."
            );

            _dashboard.Show();
            _dashboard.WindowState = WindowState.Normal;
            _dashboard.Activate();
        }
        catch (Exception ex)
        {
            try { File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash.log"), ex.ToString()); } catch { }
            System.Windows.MessageBox.Show(
                $"Failed to start TypoFormer: {ex.Message}",
                "TypoFormer Error",
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

        if (ct.IsCancellationRequested) return;

        // 1. High-Priority: Check Grammar Rule Engine (Mid-sentence caps, contractions, informal words, common typos)
        var ruleEdits = GrammarRuleEngine.FindRuleEdits(activeSentence);

        // 2. Check Neural GEC Engine
        var gecResult = _engine.CorrectSentence(activeSentence, minEditProb: 0.40f, editMargin: 0.04f);

        // Combine rule edits and neural GEC edits without duplicate keys
        var combinedEdits = new System.Collections.Generic.List<(string Original, string Replacement, string Action)>();
        foreach (var re in ruleEdits)
        {
            if (!combinedEdits.Any(e => e.Original.Equals(re.Original, StringComparison.OrdinalIgnoreCase)))
            {
                combinedEdits.Add((re.Original, re.Replacement, re.ActionDescription));
            }
        }

        if (gecResult.HasCorrection && gecResult.Edits.Count > 0)
        {
            foreach (var ge in gecResult.Edits)
            {
                string o = ge.Original.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                string r = ge.Replacement.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                if (!string.IsNullOrEmpty(o) && !string.IsNullOrEmpty(r) && !o.Equals(r, StringComparison.OrdinalIgnoreCase))
                {
                    if (!combinedEdits.Any(e => e.Original.Equals(o, StringComparison.OrdinalIgnoreCase)))
                    {
                        combinedEdits.Add((o, r, "Correct grammar"));
                    }
                }
            }
        }

        if (combinedEdits.Count == 0)
        {
            // No error detected: ensure overlay remains hidden so no intrusive UI tab pops up
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        if (ct.IsCancellationRequested) return;

        // Single word correction: target ONLY the mistyped word at its exact position
        if (combinedEdits.Count == 1)
        {
            var edit = combinedEdits[0];
            string orig = edit.Original;
            string repl = edit.Replacement;

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
        else
        {
            // Multi-error sentence correction: apply all fixes cleanly
            var ruleList = combinedEdits.Select(e => new RuleEdit { Original = e.Original, Replacement = e.Replacement }).ToList();
            string correctedSentence = GrammarRuleEngine.ApplyEdits(activeSentence, ruleList);
            string displayLabel = correctedSentence.Length > 36
                ? $"Fix {combinedEdits.Count} errors: {combinedEdits[0].Original} → {combinedEdits[0].Replacement}"
                : $"Fix: {correctedSentence}";

            _overlay.ShowSuggestion(
                screenPos: caret.ScreenPosition,
                heightOffset: caret.CaretHeight,
                displayLabel: displayLabel,
                insertionText: correctedSentence,
                badgeType: SuggestionBadgeType.Fix,
                backspaces: activeSentence.Length,
                isSelectionReplacement: false,
                isAbovePosition: false
            );
            return;
        }
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

            // 1. Run rule engine
            var ruleEdits = GrammarRuleEngine.FindRuleEdits(selectedText);

            // 2. Run GEC inference
            var gecResult = _engine.CorrectSentence(selectedText, minEditProb: 0.40f, editMargin: 0.04f);

            var combinedEdits = new System.Collections.Generic.List<(string Original, string Replacement)>();
            foreach (var re in ruleEdits)
            {
                if (!combinedEdits.Any(x => x.Original.Equals(re.Original, StringComparison.OrdinalIgnoreCase)))
                {
                    combinedEdits.Add((re.Original, re.Replacement));
                }
            }

            if (gecResult.HasCorrection && gecResult.Edits.Count > 0)
            {
                foreach (var ge in gecResult.Edits)
                {
                    string o = ge.Original.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                    string r = ge.Replacement.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                    if (!string.IsNullOrEmpty(o) && !string.IsNullOrEmpty(r) && !o.Equals(r, StringComparison.OrdinalIgnoreCase))
                    {
                        if (!combinedEdits.Any(x => x.Original.Equals(o, StringComparison.OrdinalIgnoreCase)))
                        {
                            combinedEdits.Add((o, r));
                        }
                    }
                }
            }

            if (combinedEdits.Count > 0)
            {
                var ruleList = combinedEdits.Select(x => new RuleEdit { Original = x.Original, Replacement = x.Replacement }).ToList();
                string correctedText = GrammarRuleEngine.ApplyEdits(selectedText, ruleList);

                string displayLabel;
                if (combinedEdits.Count == 1)
                {
                    displayLabel = $"{combinedEdits[0].Original} → {combinedEdits[0].Replacement}";
                }
                else
                {
                    displayLabel = correctedText.Length > 34 ? correctedText.Substring(0, 31) + "..." : correctedText;
                }

                _overlay.ShowSuggestion(
                    screenPos: e.ScreenPosition,
                    heightOffset: 0,
                    displayLabel: displayLabel,
                    insertionText: correctedText,
                    badgeType: SuggestionBadgeType.Fix,
                    backspaces: 0,
                    isSelectionReplacement: true,
                    isAbovePosition: true
                );
            }
            else
            {
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
        try { _dashboard?.Close(); } catch { }
        base.OnExit(e);
    }

    private static string FindWorkspaceRoot()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;

        // 1. If self-contained models and tokenizers exist right next to the executable, use BaseDirectory
        if ((File.Exists(Path.Combine(baseDir, @"export\lm_stage2_int8.onnx")) || File.Exists(Path.Combine(baseDir, @"export\lm_stage1_int8.onnx"))) &&
            File.Exists(Path.Combine(baseDir, @"data\tokenizer\vocab.json")))
        {
            return baseDir;
        }

        // 2. Search upwards from BaseDirectory for repository root containing pyproject.toml
        var dir = new DirectoryInfo(baseDir);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml")) &&
                Directory.Exists(Path.Combine(dir.FullName, "export")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        // 3. Current working directory
        string current = Directory.GetCurrentDirectory();
        if (File.Exists(Path.Combine(current, "pyproject.toml")) &&
            Directory.Exists(Path.Combine(current, "export")))
        {
            return current;
        }

        string desktopPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TyPoTaKer");
        if (File.Exists(Path.Combine(desktopPath, "pyproject.toml")))
            return desktopPath;

        return baseDir;
    }
}
