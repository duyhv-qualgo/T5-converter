// T5TranslatorStateful.swift
// Sample integration for the LiteRT stateful-KV-cache T5 model on iOS.
//
// SPM / CocoaPods dependency:
//   pod 'TensorFlowLiteSwift'   (or 'LiteRT' from Google AI Edge)
//
// Requires bridging to the TFLite C API for signature runner support.
// Add a bridging header with:
//   #import "tensorflow/lite/c/c_api.h"
//   #import "tensorflow/lite/c/c_api_experimental.h"
//
// Bundle resources (from litert/output/):
//   t5_mini_stateful_enc_fp32.tflite
//   t5_mini_stateful_dec_fp32.tflite   (or _dec_int8)
//
// Model signatures
// ─────────────────────────────────────────────────────────────────────────────
//  Encoder  "encode"  inputs : input_ids  int32  [1, 128]      padded token ids
//                              input_pos  int32  [1, 128]      0…127
//                              pad_mask   float  [128]         0 or -inf
//                    outputs : output_0   float  [1, 128, 384] encoder hidden states
//
//  Decoder  "decode"  inputs : encoder_hidden_states  float [1, 128, 384]
//                              decoder_input_ids      int32 [1, 1]   current token
//                              step                   int32 scalar   decode step
//                              pad_mask               float [128]
//                    outputs : output_0  float [1, 1, 20008]  logits
//
// KV cache is stored internally in the decoder interpreter's variables.
// To reset it between sentences, reload the decoder interpreter — this is
// the only correct way to clear the state without recompiling the model.

import Foundation

// MARK: - Constants  (must match litert/stateful/convert.py)

private enum T5 {
    static let seqLen:    Int32 = 128
    static let nLayers:   Int   = 4
    static let dModel:    Int32 = 384
    static let vocabSize: Int32 = 20_008
    static let eosId:     Int32 = 1
    static let padId:     Int32 = 0
    static let taskEnVi:  Int32 = 20_000
    static let taskViEn:  Int32 = 20_001
}

// MARK: - Direction

enum TranslationDirection { case enToVi, viToEn
    var taskId: Int32 { self == .enToVi ? T5.taskEnVi : T5.taskViEn }
}

// MARK: - LiteRT Signature Runner wrapper
// (same as in T5TranslatorExplicit.swift — extract to a shared file in production)

final class LiteRTSignatureRunner {

    private let interp: OpaquePointer   // TFLiteInterpreter *
    private let runner: OpaquePointer   // TFLiteSignatureRunner *

    init(interpreter: OpaquePointer, signature: String) throws {
        self.interp = interpreter
        guard let r = TFLiteInterpreterGetSignatureRunner(interpreter, signature) else {
            throw NSError(domain: "LiteRT", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "Signature '\(signature)' not found"])
        }
        self.runner = r
        guard TFLiteSignatureRunnerAllocateTensors(r) == kTfLiteOk else {
            throw NSError(domain: "LiteRT", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "AllocateTensors failed"])
        }
    }

    deinit { TFLiteSignatureRunnerDelete(runner) }

    func setInput<T>(_ name: String, values: [T]) {
        guard let tensor = TFLiteSignatureRunnerGetInputTensor(runner, name) else { return }
        var copy = values
        TFLiteTensorCopyFromBuffer(tensor, &copy, copy.count * MemoryLayout<T>.stride)
    }

    func invoke() throws {
        guard TFLiteSignatureRunnerInvoke(runner) == kTfLiteOk else {
            throw NSError(domain: "LiteRT", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Invoke failed"])
        }
    }

    func getOutputFloats(_ name: String) -> [Float] {
        guard let tensor = TFLiteSignatureRunnerGetOutputTensor(runner, name) else { return [] }
        let count  = TFLiteTensorByteSize(tensor) / MemoryLayout<Float>.stride
        var result = [Float](repeating: 0, count: count)
        TFLiteTensorCopyToBuffer(tensor, &result, TFLiteTensorByteSize(tensor))
        return result
    }
}

// MARK: - Interpreter factory

private func makeInterpreter(_ resourceName: String) throws -> OpaquePointer {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: "tflite") else {
        throw NSError(domain: "T5Stateful", code: 1,
                      userInfo: [NSLocalizedDescriptionKey:
                                 "\(resourceName).tflite not in bundle"])
    }
    let opts = TFLiteInterpreterOptionsCreate()!
    TFLiteInterpreterOptionsSetNumThreads(opts, 2)
    guard let interp = TFLiteInterpreterCreate(url.path, opts) else {
        throw NSError(domain: "T5Stateful", code: 2,
                      userInfo: [NSLocalizedDescriptionKey:
                                 "Failed to create interpreter for \(resourceName)"])
    }
    TFLiteInterpreterOptionsDelete(opts)
    return interp
}

// MARK: - Translator

/// LiteRT stateful KV-cache T5 translator.
///
/// The encoder interpreter is reused across sentences (it is stateless).
/// The decoder interpreter is reloaded per sentence to reset the internal KV cache.
///
/// Declare as `actor` so inference never blocks the main thread.
actor T5TranslatorStateful {

    private let encInterp:    OpaquePointer  // TFLiteInterpreter * (encoder, persistent)
    private let decModelName: String         // resource name without extension
    private let tokenizer:    T5Tokenizer    // your SentencePiece/BPE tokenizer wrapper

    init(decoderModelName: String = "t5_mini_stateful_dec_fp32") throws {
        encInterp    = try makeInterpreter("t5_mini_stateful_enc_fp32")
        decModelName = decoderModelName

        guard let tokURL = Bundle.main.url(forResource: "tokenizer",
                                           withExtension: "json") else {
            throw NSError(domain: "T5Stateful", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "tokenizer.json not in bundle"])
        }
        tokenizer = try T5Tokenizer(url: tokURL)
    }

    deinit { TFLiteInterpreterDelete(encInterp) }

    // MARK: Translate

    /// Returns `(translation, elapsed_ms)`.
    func translate(_ text: String, direction: TranslationDirection,
                   maxTokens: Int = 64) throws -> (String, Double) {
        let t0 = Date()

        // ── Build padded input ──────────────────────────────────────────────
        var ids = [direction.taskId] + tokenizer.encode(text).map { Int32($0) }
        if ids.last != T5.eosId { ids.append(T5.eosId) }
        let realLen = min(ids.count, Int(T5.seqLen))

        var inputIds = [Int32](repeating: 0,         count: Int(T5.seqLen))
        let inputPos = [Int32](0..<Int(T5.seqLen))
        var padMask  = [Float](repeating: 0,         count: Int(T5.seqLen))
        inputIds[0..<realLen] = ids[0..<realLen][...]
        if realLen < Int(T5.seqLen) {
            for i in realLen..<Int(T5.seqLen) { padMask[i] = -.infinity }
        }

        // ── Encode ──────────────────────────────────────────────────────────
        let encRunner = try LiteRTSignatureRunner(interpreter: encInterp, signature: "encode")
        encRunner.setInput("input_ids",  values: inputIds)
        encRunner.setInput("input_pos",  values: inputPos)
        encRunner.setInput("pad_mask",   values: padMask)
        try encRunner.invoke()
        let hiddenStates = encRunner.getOutputFloats("output_0")
        // hiddenStates shape: [1, 128, 384] = 49 152 floats

        // ── Reload decoder to reset internal KV cache ───────────────────────
        // This is required: TFLite stateful variables persist inside the
        // interpreter, so the only correct reset is to create a new interpreter.
        let decInterp = try makeInterpreter(decModelName)
        defer { TFLiteInterpreterDelete(decInterp) }

        // ── Decode loop ─────────────────────────────────────────────────────
        // Re-use one signature runner across steps — the interpreter's internal
        // state advances automatically each time Invoke() is called.
        let decRunner = try LiteRTSignatureRunner(interpreter: decInterp, signature: "decode")

        var generated:    [Int32] = []
        var currentToken: Int32   = T5.padId

        for step in 0..<maxTokens {
            decRunner.setInput("encoder_hidden_states", values: hiddenStates)
            decRunner.setInput("decoder_input_ids",     values: [currentToken])
            decRunner.setInput("step",                  values: [Int32(step)])
            decRunner.setInput("pad_mask",              values: padMask)
            try decRunner.invoke()

            let logits    = decRunner.getOutputFloats("output_0")
            // logits shape: [1, 1, 20008]
            let nextToken = Int32(argmax(logits))
            if nextToken == T5.eosId { break }
            generated.append(nextToken)
            currentToken = nextToken
        }

        let elapsedMs = Date().timeIntervalSince(t0) * 1_000
        return (tokenizer.decode(generated.map { Int($0) }), elapsedMs)
    }

    // MARK: Helpers

    private func argmax(_ v: [Float]) -> Int {
        v.indices.max(by: { v[$0] < v[$1] }) ?? 0
    }
}
