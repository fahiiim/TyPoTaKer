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

    public List<AutocompleteSuggestion> GetAutocomplete(string text, int topK = 3)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new List<AutocompleteSuggestion>();

        var (tokenIds, tokenPieces) = _tokenizer.Encode(text, addBos: true, addEos: false);
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

        return probs
            .Select((p, idx) => (p, idx))
            .Where(x => x.idx > 3) // skip special tokens
            .OrderByDescending(x => x.p)
            .Take(topK)
            .Select(x => new AutocompleteSuggestion
            {
                Text = _tokenizer.Decode(new[] { x.idx }),
                Probability = x.p,
                TokenId = x.idx
            })
            .ToList();
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

        var (tokenIds, tokenPieces) = _tokenizer.Encode(text, addBos: true, addEos: true);
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
        var edits = new List<(string Original, string Replacement, float Confidence)>();

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

            if (bestLabel == 1) // DELETE
            {
                edits.Add((_tokenizer.Decode(new[] { tid }), string.Empty, bestProb));
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
                edits.Add((_tokenizer.Decode(new[] { tid }), _tokenizer.Decode(new[] { targetToken }), bestProb));
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
                edits.Add((_tokenizer.Decode(new[] { tid }), _tokenizer.Decode(new[] { targetToken }), bestProb));
            }
            else if (bestLabel >= 4 && _idToReplace.TryGetValue(bestLabel, out int repId))
            {
                targetToken = repId;
                edits.Add((_tokenizer.Decode(new[] { tid }), _tokenizer.Decode(new[] { repId }), bestProb));
            }

            finalIds.Add(targetToken);

            if (bestLabel >= 4 && _idToAppend.TryGetValue(bestLabel, out int appId))
            {
                finalIds.Add(appId);
                edits.Add((string.Empty, _tokenizer.Decode(new[] { appId }), bestProb));
            }
        }

        string corrected = _tokenizer.Decode(finalIds).Trim();
        bool hasCorrection = edits.Count > 0 && !string.Equals(corrected, text.Trim(), StringComparison.Ordinal);

        if (!hasCorrection)
        {
            var hybridRescore = TryRescoreWithQwertyLm(text);
            if (hybridRescore != null)
            {
                return hybridRescore;
            }
        }

        return new CorrectionResult
        {
            HasCorrection = hasCorrection,
            OriginalText = text,
            CorrectedText = corrected,
            Edits = edits
        };
    }

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

    public static HashSet<string> GenerateCandidates(string word)
    {
        var cands = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string w = word.ToLowerInvariant();
        if (w.Length == 0) return cands;

        // 1. QWERTY neighbor substitution
        for (int i = 0; i < w.Length; i++)
        {
            char c = w[i];
            if (QwertyNeighbors.TryGetValue(c, out string? neighbors))
            {
                foreach (char n in neighbors)
                {
                    cands.Add(w.Substring(0, i) + n + w.Substring(i + 1));
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
            cands.Add(new string(arr));
        }

        // 3. 1-character deletion
        for (int i = 0; i < w.Length; i++)
        {
            cands.Add(w.Substring(0, i) + w.Substring(i + 1));
        }

        // 4. Common suffixes for incomplete words
        string[] suffixes = { "n", "ing", "ed", "s", "e", "d", "t" };
        foreach (var s in suffixes)
        {
            cands.Add(w + s);
        }

        cands.Remove(w);
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

        float baseNll = CalculateSentenceNll(text);
        float bestNll = baseNll;
        int bestIdx = -1;
        string? bestOrig = null;
        string? bestReplacement = null;

        for (int i = 0; i < rawTokens.Length; i++)
        {
            string raw = rawTokens[i];
            string clean = raw.Trim(".,!?:;\"'()[]{}".ToCharArray());
            if (clean.Length < 2) continue;

            var candidates = GenerateCandidates(clean);
            foreach (var cand in candidates)
            {
                if (string.Equals(cand, clean, StringComparison.OrdinalIgnoreCase)) continue;

                // Match original casing
                string casedCand = char.IsUpper(clean[0])
                    ? char.ToUpperInvariant(cand[0]) + (cand.Length > 1 ? cand.Substring(1) : "")
                    : cand;

                int startPunct = 0;
                while (startPunct < raw.Length && char.IsPunctuation(raw[startPunct])) startPunct++;
                string leading = startPunct > 0 ? raw.Substring(0, startPunct) : "";

                int endPunct = raw.Length - 1;
                while (endPunct >= startPunct && char.IsPunctuation(raw[endPunct])) endPunct--;
                string trailing = (endPunct < raw.Length - 1) ? raw.Substring(endPunct + 1) : "";

                string replacementToken = leading + casedCand + trailing;

                string[] candidateTokens = (string[])rawTokens.Clone();
                candidateTokens[i] = replacementToken;
                string candidateSentence = string.Join(" ", candidateTokens);

                float candNll = CalculateSentenceNll(candidateSentence);

                // Significant threshold: at least 3.5 nats improvement (33x more probable)
                if (candNll < bestNll - 3.5f)
                {
                    bestNll = candNll;
                    bestIdx = i;
                    bestOrig = clean;
                    bestReplacement = casedCand;
                }
            }
        }

        if (bestIdx >= 0 && bestOrig != null && bestReplacement != null)
        {
            string[] finalTokens = (string[])rawTokens.Clone();
            int startPunct = 0;
            string raw = rawTokens[bestIdx];
            while (startPunct < raw.Length && char.IsPunctuation(raw[startPunct])) startPunct++;
            string leading = startPunct > 0 ? raw.Substring(0, startPunct) : "";

            int endPunct = raw.Length - 1;
            while (endPunct >= startPunct && char.IsPunctuation(raw[endPunct])) endPunct--;
            string trailing = (endPunct < raw.Length - 1) ? raw.Substring(endPunct + 1) : "";

            finalTokens[bestIdx] = leading + bestReplacement + trailing;
            string correctedText = string.Join(" ", finalTokens);

            float conf = 1.0f - MathF.Exp(-(baseNll - bestNll));

            return new CorrectionResult
            {
                HasCorrection = true,
                OriginalText = text,
                CorrectedText = correctedText,
                Edits = new List<(string Original, string Replacement, float Confidence)>
                {
                    (bestOrig, bestReplacement, conf)
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
