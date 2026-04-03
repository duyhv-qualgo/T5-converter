// T5TranslatorExplicit.swift
// Sample integration for the LiteRT explicit-KV-cache T5 model on iOS.
//
// SPM / CocoaPods dependency:
//   pod 'TensorFlowLiteSwift'   (or 'LiteRT' from Google AI Edge)
//
// Requires bridging to the TFLite C API for signature runner support.
// Add a bridging header with:
//   #import "tensorflow/lite/c/c_api.h"
//   #import "tensorflow/lite/c/c_api_experimental.h"
//
// Bundle resource (from litert/output/):
//   t5_mini_explicit_fp32.tflite   (or _int8)
//
// Model signatures
// ─────────────────────────────────────────────────────────────────────────────
//  "encode"  inputs : args_0  int32  [1, 128]      input_ids
//                     args_1  int32  [1, 128]      input_pos (0…127)
//                     args_2  float  [128]         pad_mask  (0 or -inf)
//            outputs: output_0 float [1, 128, 384] encoder hidden states
//
//  "decode"  inputs : args_0  float  [1, 128, 384] encoder hidden states
//                     args_1  int32  [1, 1]         current token id
//                     args_2  int32  [1]             decode step index
//                     args_3  float  [128]           pad_mask
//                     args_4…11 float [1, 128, 8, 32] KV cache (4 layers × 2)
//            outputs: output_0 float [1, 1, 20008]  logits
//                     output_1…8 float [1,128,8,32]  updated KV cache

import Foundation

// MARK: - Constants

private enum T5 {
    static let seqLen:     Int32 = 128
    static let nLayers:    Int   = 4
    static let nHeads:     Int32 = 8
    static let dKV:        Int32 = 32
    static let dModel:     Int32 = 384
    static let vocabSize:  Int32 = 20_008
    static let eosId:      Int32 = 1
    static let padId:      Int32 = 0
    static let taskEnVi:   Int32 = 20_000
    static let taskViEn:   Int32 = 20_001
}

// MARK: - Direction

enum TranslationDirection { case enToVi, viToEn
    var taskId: Int32 { self == .enToVi ? T5.taskEnVi : T5.taskViEn }
}

// MARK: - LiteRT Signature Runner wrapper

/// Thin Swift wrapper around the TFLite C API signature runner.
/// Assumes the TFLite C headers are accessible via the bridging header.
final class LiteRTSignatureRunner {

    // TFLiteInterpreter and TFLiteSignatureRunner are C opaque pointers.
    // Cast from the values returned by the C API.
    private let interp:  OpaquePointer    // TFLiteInterpreter *
    private let runner:  OpaquePointer    // TFLiteSignatureRunner *

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

    /// Copy `values` into the named input tensor.
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

    /// Copy float32 output tensor into a new array.
    func getOutputFloats(_ name: String) -> [Float] {
        guard let tensor = TFLiteSignatureRunnerGetOutputTensor(runner, name) else { return [] }
        let count  = TFLiteTensorByteSize(tensor) / MemoryLayout<Float>.stride
        var result = [Float](repeating: 0, count: count)
        TFLiteTensorCopyToBuffer(tensor, &result, TFLiteTensorByteSize(tensor))
        return result
    }

    func getOutputInt32(_ name: String) -> [Int32] {
        guard let tensor = TFLiteSignatureRunnerGetOutputTensor(runner, name) else { return [] }
        let count  = TFLiteTensorByteSize(tensor) / MemoryLayout<Int32>.stride
        var result = [Int32](repeating: 0, count: count)
        TFLiteTensorCopyToBuffer(tensor, &result, TFLiteTensorByteSize(tensor))
        return result
    }
}

// MARK: - Translator

/// LiteRT explicit KV-cache T5 translator.
/// Declare as `actor` so inference never blocks the main thread.
actor T5TranslatorExplicit {

    private let interpPtr:   OpaquePointer  // TFLiteInterpreter *
    private let tokenizer:   T5Tokenizer    // your SentencePiece/BPE tokenizer wrapper

    // Preallocated KV cache buffer — reused across decode steps.
    // Shape per tensor: [1, seqLen, nHeads, dKV]
    private let kvSize:  Int
    private var kvCache: [[Float]]   // [nLayers * 2] arrays, each length kvSize

    init() throws {
        guard let modelURL = Bundle.main.url(forResource: "t5_mini_explicit_fp32",
                                             withExtension: "tflite") else {
            throw NSError(domain: "T5Explicit", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "t5_mini_explicit_fp32.tflite not in bundle"])
        }

        // Build TFLiteInterpreterOptions and create interpreter
        let opts = TFLiteInterpreterOptionsCreate()!
        TFLiteInterpreterOptionsSetNumThreads(opts, 2)
        guard let interp = TFLiteInterpreterCreate(modelURL.path, opts) else {
            throw NSError(domain: "T5Explicit", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to create interpreter"])
        }
        TFLiteInterpreterOptionsDelete(opts)
        self.interpPtr = interp

        guard let tokURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw NSError(domain: "T5Explicit", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer.json not in bundle"])
        }
        tokenizer = try T5Tokenizer(url: tokURL)

        kvSize  = Int(T5.seqLen) * Int(T5.nHeads) * Int(T5.dKV)  // 128×8×32 = 32 768
        kvCache = [[Float]](repeating: [Float](repeating: 0, count: kvSize),
                            count: T5.nLayers * 2)
    }

    // MARK: Translate

    /// Returns `(translation, elapsed_ms)`.
    func translate(_ text: String, direction: TranslationDirection,
                   maxTokens: Int = 64) throws -> (String, Double) {
        let t0 = Date()

        // ── Build padded input ──────────────────────────────────────────────
        var ids = [direction.taskId] + tokenizer.encode(text).map { Int32($0) }
        if ids.last != T5.eosId { ids.append(T5.eosId) }
        let realLen = min(ids.count, Int(T5.seqLen))

        var inputIds = [Int32](repeating: 0,           count: Int(T5.seqLen))
        var inputPos = [Int32](0..<Int(T5.seqLen))
        var padMask  = [Float](repeating: 0,           count: Int(T5.seqLen))
        inputIds[0..<realLen] = ids[0..<realLen][...]
        if realLen < Int(T5.seqLen) {
            for i in realLen..<Int(T5.seqLen) { padMask[i] = -.infinity }
        }

        // ── Encode ──────────────────────────────────────────────────────────
        let encRunner = try LiteRTSignatureRunner(interpreter: interpPtr, signature: "encode")
        encRunner.setInput("args_0", values: inputIds)
        encRunner.setInput("args_1", values: inputPos)
        encRunner.setInput("args_2", values: padMask)
        try encRunner.invoke()
        let hiddenStates = encRunner.getOutputFloats("output_0")
        // hiddenStates shape: [1, 128, 384] = 49 152 floats

        // ── Reset KV cache ──────────────────────────────────────────────────
        for i in 0..<kvCache.count { kvCache[i] = [Float](repeating: 0, count: kvSize) }

        // ── Decode loop ─────────────────────────────────────────────────────
        var generated: [Int32] = []
        var currentToken: Int32 = T5.padId

        for step in 0..<maxTokens {
            let decRunner = try LiteRTSignatureRunner(interpreter: interpPtr, signature: "decode")

            decRunner.setInput("args_0", values: hiddenStates)
            decRunner.setInput("args_1", values: [currentToken])
            decRunner.setInput("args_2", values: [Int32(step)])
            decRunner.setInput("args_3", values: padMask)
            for i in 0..<(T5.nLayers * 2) {
                decRunner.setInput("args_\(4 + i)", values: kvCache[i])
            }

            try decRunner.invoke()

            // Greedy next token
            let logits   = decRunner.getOutputFloats("output_0")
            // logits shape: [1, 1, 20008]
            let nextToken = Int32(argmax(logits))
            if nextToken == T5.eosId { break }
            generated.append(nextToken)
            currentToken = nextToken

            // Pull updated KV cache
            for i in 0..<(T5.nLayers * 2) {
                kvCache[i] = decRunner.getOutputFloats("output_\(1 + i)")
            }
        }

        let elapsedMs = Date().timeIntervalSince(t0) * 1_000
        return (tokenizer.decode(generated.map { Int($0) }), elapsedMs)
    }

    // MARK: Helpers

    private func argmax(_ v: [Float]) -> Int {
        v.indices.max(by: { v[$0] < v[$1] }) ?? 0
    }
}
