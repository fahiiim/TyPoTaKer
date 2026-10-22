using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TyPoTaKer.Desktop.Engine;

public sealed class AutocompleteSuggestion
{
    public required string Text { get; init; }
    public required float Probability { get; init; }
    public required int TokenId { get; init; }
}

public sealed class CorrectionResult
{
    public required bool HasCorrection { get; init; }
    public required string OriginalText { get; init; }
    public required string CorrectedText { get; init; }
    public required List<(string Original, string Replacement, float Confidence)> Edits { get; init; }
}

public sealed class TyPoInferenceEngine : IDisposable
{
    private readonly TyPoTokenizer _tokenizer;
    private readonly InferenceSession _lmSession;
    private readonly InferenceSession _gecSession;
    private readonly Dictionary<int, int> _idToReplace = new();
    private readonly Dictionary<int, int> _idToAppend = new();
    private readonly HashSet<string> _validWords = new(StringComparer.OrdinalIgnoreCase);

    private static readonly HashSet<string> ExcludedCompletions = new(StringComparer.OrdinalIgnoreCase)
    {
        "iv", "ix", "vi", "vii", "viii", "xi", "xii", "xiii", "xiv", "xv", "xvi", "ii", "iii", "v", "x"
    };

    private static readonly Dictionary<char, string> QwertyNeighbors = new()
    {
        ['a'] = "qwsz",
        ['b'] = "vghn",
        ['c'] = "xdfv",
        ['d'] = "serfcx",
        ['e'] = "wsdr",
        ['f'] = "drtgvc",
        ['g'] = "ftyhbvj",
        ['h'] = "gyujnb",
        ['i'] = "ujko",
        ['j'] = "huikmng",
        ['k'] = "jiolm",
        ['l'] = "kop",
        ['m'] = "njk",
        ['n'] = "bhjm",
        ['o'] = "iklp",
        ['p'] = "ol",
        ['q'] = "wa",
        ['r'] = "edft",
        ['s'] = "awedxz",
        ['t'] = "rfgy",
        ['u'] = "yhj",
        ['v'] = "cfgb",
        ['w'] = "qase",
        ['x'] = "zsdc",
        ['y'] = "tghu",
        ['z'] = "asx"
    };

    public TyPoInferenceEngine(
        string vocabPath,
        string mergesPath,
        string lmModelPath,
        string gecModelPath,
        string editMapsPath)
    {
        _tokenizer = new TyPoTokenizer(vocabPath, mergesPath);

        var sessOpts = new SessionOptions
        {
            IntraOpNumThreads = Math.Min(4, Environment.ProcessorCount)
        };

        _lmSession = new InferenceSession(lmModelPath, sessOpts);
        _gecSession = new InferenceSession(gecModelPath, sessOpts);

        // Populate valid English words from vocabulary
        if (File.Exists(vocabPath))
        {
            try
            {
                var vocabJson = JsonDocument.Parse(File.ReadAllText(vocabPath));
                foreach (var prop in vocabJson.RootElement.EnumerateObject())
                {
                    string key = prop.Name;
                    string clean = key.Replace("Ġ", "").Replace("\u0120", "").Trim().ToLowerInvariant();
                    if (clean.Length > 1 && clean.All(char.IsLetter))
                    {
                        _validWords.Add(clean);
                    }
                }
            }
            catch { }
        }

        // Add curated informal words, contractions, modern tech terms, and slang
        string[] extraWords = new[]
        {
            "wanna", "gonna", "gotta", "kinda", "sorta", "dunno", "lemme", "gimme",
            "boyfriend", "girlfriend", "online", "offline", "website", "super", "cool",
            "ok", "okay", "yeah", "yep", "nope", "idk", "imo", "tbh", "btw", "fyi",
            "pls", "thx", "msg", "txt", "sync", "wifi", "login", "logout", "signup", "signin",
            "setup", "plugin", "plugins", "subagent", "subagents", "repo", "repos",
            "github", "reddit", "youtube", "google", "chatgpt", "typo", "typotaker",
            "hey", "hi", "hello", "bye", "bro", "dude", "guy", "guys", "thanks", "welcome",
            "sorry", "please", "cant", "dont", "wont", "isnt", "arent", "wasnt", "werent",
            "hasnt", "havent", "hadnt", "doesnt", "didnt", "couldnt", "shouldnt", "wouldnt",
            "ain't", "cause", "cuz", "shoulder", "shoulders"
        };
        foreach (var w in extraWords) _validWords.Add(w);

        if (File.Exists(editMapsPath))
        {
            var mapsJson = JsonDocument.Parse(File.ReadAllText(editMapsPath));
            if (mapsJson.RootElement.TryGetProperty("id_to_replace", out var rep))
            {
                foreach (var prop in rep.EnumerateObject())
                    _idToReplace[int.Parse(prop.Name)] = prop.Value.GetInt32();
            }
            if (mapsJson.RootElement.TryGetProperty("id_to_append", out var app))
            {
                foreach (var prop in app.EnumerateObject())
                    _idToAppend[int.Parse(prop.Name)] = prop.Value.GetInt32();
            }
        }
    }

    public bool IsValidWord(string word)
    {
        if (string.IsNullOrWhiteSpace(word)) return true;
        string clean = word.Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray()).ToLowerInvariant();
        if (clean.Length == 0) return true;
        if (clean.Length == 1) return clean == "a" || clean == "i";
        return _validWords.Contains(clean);
    }

    public List<AutocompleteSuggestion> GetAutocomplete(string text, int topK = 3)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new List<AutocompleteSuggestion>();

        // Always trim trailing whitespace for next-word completion context
        string trimmed = text.TrimEnd();
        if (string.IsNullOrWhiteSpace(trimmed))
            return new List<AutocompleteSuggestion>();

        var (tokenIds, tokenPieces) = _tokenizer.Encode(trimmed, addBos: true, addEos: false);
        int seqLen = tokenIds.Count;
        if (seqLen < 2) return new List<AutocompleteSuggestion>();

        var charArray = _tokenizer.BuildCharTensor(tokenPieces);
        var charTensor = new DenseTensor<long>(new[] { 1, seqLen, _tokenizer.MaxCharsPerToken });
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < _tokenizer.MaxCharsPerToken; c++)
                charTensor[0, t, c] = charArray[0, t, c];
        }

        var tokTensor = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, seqLen });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("token_ids", tokTensor),
            NamedOnnxValue.CreateFromTensor("char_ids", charTensor)
        };

        using var results = _lmSession.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();

        int vocabSize = logits.Dimensions[2];
        var lastLogits = new float[vocabSize];
        float maxLogit = float.MinValue;

        for (int v = 0; v < vocabSize; v++)
        {
            float val = logits[0, seqLen - 1, v];
            lastLogits[v] = val;
            if (val > maxLogit) maxLogit = val;
        }

        float sumExp = 0f;
        var probs = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++)
        {
            probs[v] = MathF.Exp(lastLogits[v] - maxLogit);
            sumExp += probs[v];
        }
        for (int v = 0; v < vocabSize; v++) probs[v] /= sumExp;

        var ranked = probs
            .Select((p, idx) => (p, idx))
            .Where(x => x.idx > 3)
            .OrderByDescending(x => x.p)
            .Take(150);

        var suggestions = new List<AutocompleteSuggestion>();

        foreach (var (p, idx) in ranked)
        {
            if (p < 0.015f) break;

            string decoded = _tokenizer.Decode(new[] { idx });
            // In BPE decoding, tokens starting with space represent whole new words
            if (!decoded.StartsWith(" "))
                continue;

            string clean = decoded.Trim();
            if (clean.Length == 0) continue;
            if (clean.Length == 1 && clean != "a" && clean != "A" && clean != "I") continue;
            if (!clean.All(char.IsLetter)) continue;
            if (ExcludedCompletions.Contains(clean)) continue;
            if (!_validWords.Contains(clean)) continue;

            suggestions.Add(new AutocompleteSuggestion
            {
                Text = clean,
                Probability = p,
                TokenId = idx
            });

            if (suggestions.Count >= topK)
                break;
        }

        return suggestions;
    }

    public CorrectionResult CorrectSentence(string text, float minEditProb = 0.65f, float editMargin = 0.08f)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new CorrectionResult
            {
                HasCorrection = false,
                OriginalText = text,
                CorrectedText = text,
                Edits = new()
            };
        }

        string trimmedText = text.Trim();
        string[] originalWords = trimmedText.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        bool allOriginalWordsValid = originalWords.All(IsValidWord);

        var (tokenIds, tokenPieces) = _tokenizer.Encode(trimmedText, addBos: true, addEos: true);
        int seqLen = tokenIds.Count;

        var charArray = _tokenizer.BuildCharTensor(tokenPieces);
        var charTensor = new DenseTensor<long>(new[] { 1, seqLen, _tokenizer.MaxCharsPerToken });
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < _tokenizer.MaxCharsPerToken; c++)
                charTensor[0, t, c] = charArray[0, t, c];
        }

        var tokTensor = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, seqLen });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("token_ids", tokTensor),
            NamedOnnxValue.CreateFromTensor("char_ids", charTensor)
        };

        using var results = _gecSession.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();

        int numLabels = logits.Dimensions[2];
        var finalIds = new List<int>();

        for (int t = 0; t < seqLen; t++)
        {
            float maxLogit = float.MinValue;
            for (int l = 0; l < numLabels; l++)
                if (logits[0, t, l] > maxLogit) maxLogit = logits[0, t, l];

            float sumExp = 0f;
            var probs = new float[numLabels];
            for (int l = 0; l < numLabels; l++)
            {
                probs[l] = MathF.Exp(logits[0, t, l] - maxLogit);
                sumExp += probs[l];
            }
            for (int l = 0; l < numLabels; l++) probs[l] /= sumExp;

            int bestLabel = 0;
            float bestProb = probs[0];
            for (int l = 1; l < numLabels; l++)
            {
                if (probs[l] > bestProb)
                {
                    bestProb = probs[l];
                    bestLabel = l;
                }
            }

            int tid = (int)tokenIds[t];
            string piece = tokenPieces[t];

            // Gated confidence threshold
            if (bestLabel != 0 && (bestProb < minEditProb || (bestProb - probs[0]) < editMargin))
            {
                bestLabel = 0;
            }

            // Sub-token protection:
            // If all words in the input text are valid words, never edit sub-tokens (tokens without leading space)
            // unless the model is extremely certain (>= 0.92f)
            if (bestLabel != 0 && t > 0 && t < seqLen - 1)
            {
                string decodedPiece = _tokenizer.Decode(new[] { tid });
                bool isSubToken = !decodedPiece.StartsWith(" ");

                if (isSubToken && allOriginalWordsValid && bestProb < 0.92f)
                {
                    bestLabel = 0; // Protect sub-token of valid word!
                }
                else if (isSubToken && bestLabel >= 4 && _idToReplace.TryGetValue(bestLabel, out int checkRepId))
                {
                    string repDecoded = _tokenizer.Decode(new[] { checkRepId });
                    // A suffix cannot be replaced with a token having a leading space (e.g. anna -> " and")
                    if (repDecoded.StartsWith(" "))
                    {
                        bestLabel = 0;
                    }
                }
            }

            if (bestLabel == 1) // DELETE
            {
                continue;
            }

            int targetToken = tid;
            if (bestLabel == 2) // CASE_CAP
            {
                string dec = _tokenizer.Decode(new[] { tid });
                if (dec.Length > 0)
                {
                    string cap = char.ToUpperInvariant(dec[0]) + (dec.Length > 1 ? dec.Substring(1) : "");
                    var (recIds, _) = _tokenizer.Encode(cap, addBos: false, addEos: false);
                    if (recIds.Count > 0) targetToken = (int)recIds[0];
                }
            }
            else if (bestLabel == 3) // CASE_LOWER
            {
                string dec = _tokenizer.Decode(new[] { tid });
                if (dec.Length > 0)
                {
                    string lower = dec.ToLowerInvariant();
                    var (recIds, _) = _tokenizer.Encode(lower, addBos: false, addEos: false);
                    if (recIds.Count > 0) targetToken = (int)recIds[0];
                }
            }
            else if (bestLabel >= 4 && _idToReplace.TryGetValue(bestLabel, out int repId))
            {
                targetToken = repId;
            }

            finalIds.Add(targetToken);

            if (bestLabel >= 4 && _idToAppend.TryGetValue(bestLabel, out int appId))
            {
                finalIds.Add(appId);
            }
        }

        string corrected = _tokenizer.Decode(finalIds).Trim();
        bool hasCorrection = !string.Equals(corrected, trimmedText, StringComparison.Ordinal);

        var edits = new List<(string Original, string Replacement, float Confidence)>();

        if (hasCorrection)
        {
            // Word-level extraction so edits are ALWAYS whole words (no "anna -> and")
            edits = ExtractWordEdits(trimmedText, corrected);

            // Filter out edits where original word was already valid but replacement is invalid
            edits = edits.Where(e =>
            {
                if (IsValidWord(e.Original) && !IsValidWord(e.Replacement)) return false;
                return !string.Equals(e.Original, e.Replacement, StringComparison.OrdinalIgnoreCase);
            }).ToList();

            if (edits.Count == 0)
            {
                hasCorrection = false;
                corrected = trimmedText;
            }
        }

        // Only run QWERTY LM rescoring if GEC found nothing AND there is an unrecognized word in the text!
        if (!hasCorrection && !allOriginalWordsValid)
        {
            var hybridRescore = TryRescoreWithQwertyLm(trimmedText);
            if (hybridRescore != null)
            {
                return hybridRescore;
            }
        }

        return new CorrectionResult
        {
            HasCorrection = hasCorrection,
            OriginalText = text,
            CorrectedText = hasCorrection ? corrected : text,
            Edits = edits
        };
    }

    private static List<(string Original, string Replacement, float Confidence)> ExtractWordEdits(string origText, string corrText)
    {
        var edits = new List<(string Original, string Replacement, float Confidence)>();
        string[] origWords = origText.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        string[] corrWords = corrText.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);

        if (origWords.Length == corrWords.Length)
        {
            for (int i = 0; i < origWords.Length; i++)
            {
                string o = origWords[i].Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                string c = corrWords[i].Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
                if (!string.Equals(o, c, StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(o) && !string.IsNullOrEmpty(c))
                {
                    edits.Add((origWords[i], corrWords[i], 0.90f));
                }
            }
        }
        else
        {
            // Sentence-level replacement
            edits.Add((origText, corrText, 0.85f));
        }

        return edits;
    }

    public List<string> GenerateCandidates(string word)
    {
        var cands = new List<string>();
        string w = word.ToLowerInvariant();
        if (w.Length == 0) return cands;

        // 1. QWERTY neighbor substitution (only 1 edit distance)
        for (int i = 0; i < w.Length; i++)
        {
            char c = w[i];
            if (QwertyNeighbors.TryGetValue(c, out string? neighbors))
            {
                foreach (char n in neighbors)
                {
                    string cand = w.Substring(0, i) + n + w.Substring(i + 1);
                    if (_validWords.Contains(cand) && !cands.Contains(cand))
                    {
                        cands.Add(cand);
                        if (cands.Count >= 6) return cands;
                    }
                }
            }
        }

        // 2. Adjacent transposition
        for (int i = 0; i < w.Length - 1; i++)
        {
            char[] arr = w.ToCharArray();
            char tmp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = tmp;
            string cand = new string(arr);
            if (_validWords.Contains(cand) && !cands.Contains(cand))
            {
                cands.Add(cand);
                if (cands.Count >= 6) return cands;
            }
        }

        // 3. 1-character deletion
        for (int i = 0; i < w.Length; i++)
        {
            string cand = w.Substring(0, i) + w.Substring(i + 1);
            if (_validWords.Contains(cand) && !cands.Contains(cand))
            {
                cands.Add(cand);
                if (cands.Count >= 6) return cands;
            }
        }

        return cands;
    }

    public float CalculateSentenceNll(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return 999f;
        var (tokenIds, tokenPieces) = _tokenizer.Encode(text, addBos: true, addEos: true);
        int seqLen = tokenIds.Count;
        if (seqLen < 2) return 999f;

        var charArray = _tokenizer.BuildCharTensor(tokenPieces);
        var charTensor = new DenseTensor<long>(new[] { 1, seqLen - 1, _tokenizer.MaxCharsPerToken });
        for (int t = 0; t < seqLen - 1; t++)
        {
            for (int c = 0; c < _tokenizer.MaxCharsPerToken; c++)
                charTensor[0, t, c] = charArray[0, t, c];
        }

        var tokSub = tokenIds.Take(seqLen - 1).ToArray();
        var tokTensor = new DenseTensor<long>(tokSub, new[] { 1, seqLen - 1 });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("token_ids", tokTensor),
            NamedOnnxValue.CreateFromTensor("char_ids", charTensor)
        };

        using var results = _lmSession.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();

        int vocabSize = logits.Dimensions[2];
        float totalNll = 0f;

        for (int t = 0; t < seqLen - 1; t++)
        {
            float maxLogit = float.MinValue;
            for (int v = 0; v < vocabSize; v++)
            {
                float val = logits[0, t, v];
                if (val > maxLogit) maxLogit = val;
            }

            float sumExp = 0f;
            for (int v = 0; v < vocabSize; v++)
            {
                sumExp += MathF.Exp(logits[0, t, v] - maxLogit);
            }

            int targetToken = (int)tokenIds[t + 1];
            float targetLogit = logits[0, t, targetToken];
            float targetProb = MathF.Exp(targetLogit - maxLogit) / sumExp;

            totalNll -= MathF.Log(Math.Max(targetProb, 1e-9f));
        }

        return totalNll;
    }

    private CorrectionResult? TryRescoreWithQwertyLm(string text)
    {
        string[] rawTokens = text.Split(' ');
        if (rawTokens.Length == 0) return null;

        // Find the FIRST word that is NOT a valid word
        int suspiciousIdx = -1;
        string? suspiciousClean = null;
        for (int i = 0; i < rawTokens.Length; i++)
        {
            string clean = rawTokens[i].Trim(".,!?:;\"'()[]{}<>-/\\".ToCharArray());
            if (clean.Length > 1 && !IsValidWord(clean))
            {
                suspiciousIdx = i;
                suspiciousClean = clean;
                break;
            }
        }

        // If all words are valid English words, no QWERTY rescoring is needed!
        if (suspiciousIdx < 0 || suspiciousClean == null) return null;

        var candidates = GenerateCandidates(suspiciousClean);
        if (candidates.Count == 0) return null;

        float baseNll = CalculateSentenceNll(text);
        float bestNll = baseNll;
        string? bestReplacement = null;

        string raw = rawTokens[suspiciousIdx];
        int startPunct = 0;
        while (startPunct < raw.Length && char.IsPunctuation(raw[startPunct])) startPunct++;
        string leading = startPunct > 0 ? raw.Substring(0, startPunct) : "";

        int endPunct = raw.Length - 1;
        while (endPunct >= startPunct && char.IsPunctuation(raw[endPunct])) endPunct--;
        string trailing = (endPunct < raw.Length - 1) ? raw.Substring(endPunct + 1) : "";

        foreach (var cand in candidates)
        {
            if (string.Equals(cand, suspiciousClean, StringComparison.OrdinalIgnoreCase)) continue;

            string casedCand = char.IsUpper(suspiciousClean[0])
                ? char.ToUpperInvariant(cand[0]) + (cand.Length > 1 ? cand.Substring(1) : "")
                : cand;

            string replacementToken = leading + casedCand + trailing;
            string[] candidateTokens = (string[])rawTokens.Clone();
            candidateTokens[suspiciousIdx] = replacementToken;
            string candidateSentence = string.Join(" ", candidateTokens);

            float candNll = CalculateSentenceNll(candidateSentence);

            if (candNll < bestNll - 2.8f) // Significant improvement
            {
                bestNll = candNll;
                bestReplacement = casedCand;
            }
        }

        if (bestReplacement != null)
        {
            string[] finalTokens = (string[])rawTokens.Clone();
            finalTokens[suspiciousIdx] = leading + bestReplacement + trailing;
            string correctedText = string.Join(" ", finalTokens);
            float conf = Math.Clamp(1.0f - MathF.Exp(-(baseNll - bestNll)), 0.60f, 0.98f);

            return new CorrectionResult
            {
                HasCorrection = true,
                OriginalText = text,
                CorrectedText = correctedText,
                Edits = new List<(string Original, string Replacement, float Confidence)>
                {
                    (suspiciousClean, bestReplacement, conf)
                }
            };
        }

        return null;
    }

    public void Dispose()
    {
        _lmSession?.Dispose();
        _gecSession?.Dispose();
    }
}
