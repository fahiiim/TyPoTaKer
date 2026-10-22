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
    private TrayManager? _tray;
    private System.Timers.Timer? _debounceTimer;
    private CancellationTokenSource? _inferenceCts;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        string root = FindWorkspaceRoot();
        string vocab = Path.Combine(root, @"data\tokenizer\vocab.json");
        string merges = Path.Combine(root, @"data\tokenizer\merges.txt");
        string lm = Path.Combine(root, @"export\lm_stage1_int8.onnx");
        string gec = Path.Combine(root, @"export\gec_stage1_int8.onnx");
        string maps = Path.Combine(root, @"data\processed\gec\v2\edit_maps.json");

        try
        {
            _engine = new TyPoInferenceEngine(vocab, merges, lm, gec, maps);
            _overlay = new CaretOverlayWindow();
            _hook = new KeyboardHook();
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
            _overlay.DismissClicked += (s, args) =>
            {
                if (_hook != null) _hook.IsDismissed = true;
                _overlay.HideOverlay();
            };

            _tray.EnabledToggled += (s, enabled) =>
            {
                if (_hook != null) _hook.IsEnabled = enabled;
                if (!enabled) _overlay.HideOverlay();
            };

            _tray.ResetRequested += (s, args) =>
            {
                _hook.ClearBuffer();
                _overlay.HideOverlay();
                _tray.ShowNotification("Buffer Reset", "Typing context buffer cleared.");
            };

            _tray.ExitRequested += (s, args) =>
            {
                Shutdown();
            };

            _hook.Start();

            _tray.ShowNotification(
                "TyPoTaKer Active",
                "Ready for typing! Press Ctrl+Space to accept suggestions. Icon running in tray."
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

        // Only offer suggestions when a word, sentence, or line has been completed
        if (!isWordBoundary)
        {
            Dispatcher.Invoke(() => _overlay?.HideOverlay());
            return;
        }

        if (ct.IsCancellationRequested) return;

        // 1. High-Priority: Check GEC grammar and typo corrections on the active sentence/phrase
        var gecResult = _engine.CorrectSentence(activeSentence, minEditProb: 0.60f, editMargin: 0.08f);
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
                            caretScreenPos: caret.ScreenPosition,
                            caretHeight: caret.CaretHeight,
                            displayLabel: displayLabel,
                            insertionText: insertion,
                            isCorrection: true,
                            backspaces: backspaces
                        );
                        return;
                    }
                }
            }
            else
            {
                // Multi-error sentence correction
                string displayLabel = $"Fix line: {gecResult.CorrectedText}";
                _overlay.ShowSuggestion(
                    caretScreenPos: caret.ScreenPosition,
                    caretHeight: caret.CaretHeight,
                    displayLabel: displayLabel,
                    insertionText: gecResult.CorrectedText,
                    isCorrection: true,
                    backspaces: activeSentence.Length
                );
                return;
            }
        }

        if (ct.IsCancellationRequested) return;

        // 2. Second-Priority: Contextual autocomplete suggestions on the completed word boundary
        var suggestions = _engine.GetAutocomplete(activeSentence, topK: 1);
        if (suggestions.Count > 0 && suggestions[0].Probability >= 0.08f)
        {
            if (ct.IsCancellationRequested) return;

            var best = suggestions[0];
            string completion = best.Text.Trim();
            if (!string.IsNullOrWhiteSpace(completion))
            {
                _overlay.ShowSuggestion(
                    caretScreenPos: caret.ScreenPosition,
                    caretHeight: caret.CaretHeight,
                    displayLabel: completion,
                    insertionText: completion,
                    isCorrection: false,
                    backspaces: 0
                );
                return;
            }
        }

        Dispatcher.Invoke(() => _overlay?.HideOverlay());
    }

    /// <summary>
    /// Extracts the current sentence or line being typed from the end of the text.
    /// </summary>
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

        // Cap to last 160 characters so we stay well within model window
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
            _overlay.HideOverlay();

            // Simulate keystrokes into active document
            KeyboardHook.SendText(insertion, backspaces);
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _debounceTimer?.Dispose();
        _inferenceCts?.Cancel();
        _inferenceCts?.Dispose();
        _hook?.Dispose();
        _tray?.Dispose();
        _engine?.Dispose();
        base.OnExit(e);
    }

    private static string FindWorkspaceRoot()
    {
        // 1. Check current working directory
        string current = Directory.GetCurrentDirectory();
        if (Directory.Exists(Path.Combine(current, "data", "tokenizer")) &&
            Directory.Exists(Path.Combine(current, "export")))
        {
            return current;
        }

        // 2. Search upwards from AppDomain BaseDirectory
        var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "data", "tokenizer")) &&
                Directory.Exists(Path.Combine(dir.FullName, "export")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        // 3. Fallback: Desktop/TyPoTaKer if known
        string desktopPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "TyPoTaKer");
        if (Directory.Exists(desktopPath))
            return desktopPath;

        return AppDomain.CurrentDomain.BaseDirectory;
    }
}
