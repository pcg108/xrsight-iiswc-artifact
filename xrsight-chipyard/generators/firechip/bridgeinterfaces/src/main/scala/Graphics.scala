// See LICENSE for license details.

package firechip.bridgeinterfaces

import chisel3._
import chisel3.util.Decoupled

// Note: This file is heavily commented as it serves as a bridge walkthrough
// example in the FireSim docs

// Note: All code in this file must be isolated from target-side generators/classes/etc
// since this is also injected into the midas compiler.

// DOC include start: UART Bridge Target-Side Interface
class GraphicsPortIO extends Bundle {

  // incoming from host to bridge
  val rx = Flipped(Decoupled(UInt(32.W)))
  // indicates that the host is currently transmitting a rutabaga stream message
  val hostTransmit = Input(Bool())

  // outgoing from bridge to host
  val tx = Decoupled(UInt(32.W))
  // indicates that the guest is currently transmitting a rutabaga stream message
  val guestTransmit = Output(Bool())

  // DMA/Streaming ports
  // val rx_stream = Flipped(Decoupled(UInt(512.W)))
  // val tx_stream = Decoupled(UInt(512.W))

  // // stream request
  // val stream_req_rx = Flipped(Decoupled(UInt(32.W)))
  // val stream_req_tx = Flipped(Decoupled(UInt(32.W)))

}

// the IO to connect bridge to target, so the IO is flipped because the output of the target is the input to the bridge, and the input to the bridge is output to the target
class GraphicsBridgeTargetIO extends Bundle {
  val clock = Input(Clock())
  val graphics = Flipped(new GraphicsPortIO)
  // Note this reset is optional and used only to reset target-state modeled
  // in the bridge. This reset is just like any other Bool included in your target
  // interface, simply appears as another Bool in the input token.
  val reset = Input(Bool())
}
// DOC include end: UART Bridge Target-Side Interface

// DOC include start: UART Bridge Constructor Arg
// Out bridge module constructor argument. This captures all of the extra
// metadata we'd like to pass to the host-side BridgeModule. Note, we need to
// use a single case class to do so, even if it is simply to wrap a primitive
// type, as is the case for the div Int.
case class GraphicsBridgeKey()
// DOC include end: UART Bridge Constructor Arg
