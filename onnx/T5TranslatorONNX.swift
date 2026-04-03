// T5TranslatorONNX.swift
// Sample integration for the ONNX-format T5 translator on iOS.
//
// Full production implementation lives at:
//   t5-translator-ios/T5TranslatorApp/T5Translator.swift
//
// Podfile dependency:
//   pod 'onnxruntime-objc'
//
// Bundle resources (from onnx/output/fp32/ or int8/):
//   encoder_model.onnx
//   decoder_model.onnx
//   decoder_with_past_model.onnx
//   tokenizer.json
//
// For INT8 quantized models use the *_quantized.onnx variants and rename
// them to the names above when adding to the Xcode bundle.

import Foundation
import OnnxRuntimeBindings

// MARK: - Constants

private enum T5 {
    static let seqLen:    Int   = 128
    static let nLayers:   Int   = 4
    static let eosId:     Int64 = 1
    static let padId:     Int64 = 0
    static let taskEnVi:  Int64 = 20_000   // "translate English to Vietnamese:"
    static let taskViEn:  Int64 = 20_001   // "translate Vietnamese to English:"
}

// MARK: - Direction

enum TranslationDirection {
    case enToVi, viToEn
    var taskId: Int64 { self == .enToVi ? T5.taskEnVi : T5.taskViEn }
}

// MARK: - Translator

/// Runs three-session ONNX inference (encoder / first-decoder / decoder-with-past).
/// Declare as `actor` so inference never blocks the main thread.
actor T5TranslatorONNX {

    private let encSess: ORTSession
    private let decSess: ORTSession
    private let dwpSess: ORTSession
    private let tokenizer: T5Tokenizer   // your SentencePiece/BPE tokenizer wrapper

    // ── Output / input name lists ────────────────────────────────────────────

    /// decoder_model outputs: logits + 4 × (decoder.key, decoder.value,
    ///                                       encoder.key, encoder.value)
    private static let decOutNames: [String] = {
        var n = ["logits"]
        for i in 0..<T5.nLayers {
            n += ["present.\(i).decoder.key",  "present.\(i).decoder.value",
                  "present.\(i).encoder.key",  "present.\(i).encoder.value"]
        }
        return n
    }()

    /// decoder_with_past outputs: logits + 4 × (decoder.key, decoder.value)
    /// Encoder KV is static — not updated after the first step.
    private static let dwpOutNames: [String] = {
        var n = ["logits"]
        for i in 0..<T5.nLayers {
            n += ["present.\(i).decoder.key", "present.\(i).decoder.value"]
        }
        return n
    }()

    /// decoder_with_past inputs: input_ids, encoder_attention_mask,
    ///                           + 4 × (decoder.key, decoder.value,
    ///                                  encoder.key, encoder.value)
    private static let dwpInNames: [String] = {
        var n = ["input_ids", "encoder_attention_mask"]
        for i in 0..<T5.nLayers {
            n += ["past_key_values.\(i).decoder.key",  "past_key_values.\(i).decoder.value",
                  "past_key_values.\(i).encoder.key",  "past_key_values.\(i).encoder.value"]
        }
        return n
    }()

    // MARK: Init

    init() throws {
        let env  = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)

        func load(_ name: String) throws -> ORTSession {
            guard let path = Bundle.main.path(forResource: name, ofType: "onnx") else {
                throw NSError(domain: "T5ONNX", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "\(name).onnx not in bundle"])
            }
            return try ORTSession(env: env, modelPath: path, sessionOptions: opts)
        }
        encSess = try load("encoder_model")
        decSess = try load("decoder_model")
        dwpSess = try load("decoder_with_past_model")

        guard let tokURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw NSError(domain: "T5ONNX", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer.json not in bundle"])
        }
        tokenizer = try T5Tokenizer(url: tokURL)
    }

    // MARK: Translate

    /// Returns `(translation, elapsed_ms)`.
    func translate(_ text: String, direction: TranslationDirection,
                   maxTokens: Int = 64) throws -> (String, Double) {
        let t0 = Date()

        // ── Tokenise input ─────────────────────────────────────────────────
        var inputIds = [direction.taskId] + tokenizer.encode(text).map { Int64($0) }
        if inputIds.last != T5.eosId { inputIds.append(T5.eosId) }
        let seqLen   = inputIds.count
        let attnMask = [Int64](repeating: 1, count: seqLen)

        // ── Encoder ────────────────────────────────────────────────────────
        let encOut = try encSess.run(
            withInputs: [
                "input_ids":      try int64Tensor(inputIds,  shape: [1, seqLen]),
                "attention_mask": try int64Tensor(attnMask,  shape: [1, seqLen]),
            ],
            outputNames: Set(["last_hidden_state"]), runOptions: nil)
        let hiddenStates = encOut["last_hidden_state"]!
        let encMaskTensor = try int64Tensor(attnMask, shape: [1, seqLen])

        // ── First decoder step (cold-start, builds full KV cache) ──────────
        let dec0Out = try decSess.run(
            withInputs: [
                "input_ids":              try int64Tensor([T5.padId], shape: [1, 1]),
                "encoder_hidden_states":  hiddenStates,
                "encoder_attention_mask": encMaskTensor,
            ],
            outputNames: Set(Self.decOutNames), runOptions: nil)

        var generated = [Int64(argmax(try floats(dec0Out["logits"]!)))]
        var kv = Dictionary(uniqueKeysWithValues:
            Self.decOutNames.dropFirst().map { ($0, dec0Out[$0]!) })

        // ── Autoregressive decode loop ─────────────────────────────────────
        for _ in 0..<(maxTokens - 1) {
            guard let lastToken = generated.last, lastToken != T5.eosId else { break }

            var inputs: [String: ORTValue] = [
                "input_ids":              try int64Tensor([lastToken], shape: [1, 1]),
                "encoder_attention_mask": encMaskTensor,
            ]
            for name in Self.dwpInNames where name.hasPrefix("past_key_values") {
                inputs[name] = kv[name.replacingOccurrences(of: "past_key_values", with: "present")]
            }

            let dwpOut   = try dwpSess.run(
                withInputs: inputs, outputNames: Set(Self.dwpOutNames), runOptions: nil)
            let nextToken = Int64(argmax(try floats(dwpOut["logits"]!)))
            if nextToken == T5.eosId { break }
            generated.append(nextToken)
            for name in Self.dwpOutNames.dropFirst() { kv[name] = dwpOut[name]! }
        }

        let elapsedMs = Date().timeIntervalSince(t0) * 1_000
        return (tokenizer.decode(generated.map { Int($0) }), elapsedMs)
    }

    // MARK: Tensor helpers

    private func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        var copy = values
        let data = NSMutableData(bytes: &copy, length: copy.count * MemoryLayout<Int64>.stride)
        return try ORTValue(tensorData: data, elementType: .int64,
                            shape: shape.map { NSNumber(value: $0) })
    }

    private func floats(_ value: ORTValue) throws -> [Float] {
        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func argmax(_ values: [Float]) -> Int {
        values.indices.max(by: { values[$0] < values[$1] }) ?? 0
    }
}
