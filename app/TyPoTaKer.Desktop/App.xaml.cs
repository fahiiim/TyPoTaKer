using System;
using System.IO;
using System.Windows;
using TyPoTaKer.Desktop.Engine;

namespace TyPoTaKer.Desktop;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        string root = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\.."));
        string vocab = Path.Combine(root, @"data\tokenizer\vocab.json");
        string merges = Path.Combine(root, @"data\tokenizer\merges.txt");
        string lm = Path.Combine(root, @"export\lm_stage1_int8.onnx");
        string gec = Path.Combine(root, @"export\gec_stage1_int8.onnx");
        string maps = Path.Combine(root, @"data\processed\gec\v2\edit_maps.json");

        Console.WriteLine("=== TyPoTaKer C# Engine Self-Test ===");
        try
        {
            using var engine = new TyPoInferenceEngine(vocab, merges, lm, gec, maps);

            // Test Autocomplete
            string prompt = "The server listens";
            var suggestions = engine.GetAutocomplete(prompt, topK: 3);
            Console.WriteLine($"LM Prompt: '{prompt}'");
            foreach (var s in suggestions)
            {
                Console.WriteLine($"  -> '{s.Text}' (prob={s.Probability:P1})");
            }

            // Test GEC
            string typo = "This is teh final document.";
            var gecResult = engine.CorrectSentence(typo);
            Console.WriteLine($"GEC Input:  '{typo}'");
            Console.WriteLine($"GEC Output: '{gecResult.CorrectedText}' (hasCorrection={gecResult.HasCorrection})");
            foreach (var edit in gecResult.Edits)
            {
                Console.WriteLine($"  Edit: '{edit.Original}' -> '{edit.Replacement}' (conf={edit.Confidence:P1})");
            }
            Console.WriteLine("=== Engine Self-Test Succeeded ===");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Engine test failed: {ex.Message}\n{ex.StackTrace}");
        }

        Shutdown();
    }
}
