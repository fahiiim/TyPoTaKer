using System;
using System.IO;
using System.Linq;
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

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        string root = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\.."));
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

            // Setup 130ms typing debouncer
            _debounceTimer = new System.Timers.Timer(130) { AutoReset = false };
            _debounceTimer.Elapsed += OnDebounceElapsed;

            _hook.TypingActivity += (s, args) =>
            {
                _debounceTimer.Stop();
                _debounceTimer.Start();
            };

            _hook.AcceptRequested += OnAcceptRequested;
            _hook.DismissRequested += (s, args) => _overlay.HideOverlay();

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
        if (_engine == null || _overlay == null || _hook == null || !_hook.IsEnabled)
            return;

        var caret = CaretTracker.GetCaretInfo();
        if (!caret.IsVisible || caret.IsPassword)
        {
            _overlay.HideOverlay();
            return;
        }

        string rawText = caret.SurroundingText ?? _hook.GetBuffer();
        if (string.IsNullOrWhiteSpace(rawText) || rawText.Trim().Length < 2)
        {
            _overlay.HideOverlay();
            return;
        }

        string text = rawText.Trim();

        // 1. Check GEC for typos on recent sentence/phrase
        var gecResult = _engine.CorrectSentence(text);
        if (gecResult.HasCorrection && gecResult.Edits.Count > 0)
        {
            var lastEdit = gecResult.Edits.Last();
            string orig = lastEdit.Original.Trim();
            string repl = lastEdit.Replacement.Trim();

            // Calculate backspaces needed for the mistyped token
            int backspaces = orig.Length;
            string displayLabel = $"{orig} → {repl}";

            _overlay.ShowSuggestion(
                caretScreenPos: caret.ScreenPosition,
                caretHeight: caret.CaretHeight,
                displayLabel: displayLabel,
                insertionText: repl,
                isCorrection: true,
                backspaces: backspaces
            );
            return;
        }

        // 2. Check LM next-word autocomplete
        var suggestions = _engine.GetAutocomplete(text, topK: 1);
        if (suggestions.Count > 0 && suggestions[0].Probability >= 0.10f)
        {
            var best = suggestions[0];
            string displayLabel = best.Text;

            _overlay.ShowSuggestion(
                caretScreenPos: caret.ScreenPosition,
                caretHeight: caret.CaretHeight,
                displayLabel: displayLabel,
                insertionText: best.Text,
                isCorrection: false,
                backspaces: 0
            );
            return;
        }

        _overlay.HideOverlay();
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
        _hook?.Dispose();
        _tray?.Dispose();
        _engine?.Dispose();
        base.OnExit(e);
    }
}
