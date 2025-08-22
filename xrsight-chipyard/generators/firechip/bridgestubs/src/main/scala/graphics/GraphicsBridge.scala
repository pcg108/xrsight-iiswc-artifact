// See LICENSE for license details

package firechip.bridgestubs

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters

import firesim.lib.bridgeutils._

import firechip.bridgeinterfaces._

// Note: This file is heavily commented as it serves as a bridge walkthrough
// example in the FireSim docs

/*
  This is the dummy target interface to connect to the bridge
  - we define an IO to connect the bridge to target RTL using the BridgeTargetIO
  - this module will get removed later and it's inputs and outputs will be wired directly to the top-level of the SoC
  - the top-level of the SoC exposes the GraphicsTL module, so the bridge will be connected to that 
*/

// DOC include start: UART Bridge Target-Side Module
class GraphicsBridge()(implicit p: Parameters) extends BlackBox
    with Bridge[HostPortIO[GraphicsBridgeTargetIO]] {
  // Module portion corresponding to this bridge
  val moduleName = "firechip.goldengateimplementations.GraphicsBridgeModule"

  // Since we're extending BlackBox this is the port will connect to in our target's RTL
  val io = IO(new GraphicsBridgeTargetIO)
  // Implement the bridgeIO member of Bridge using HostPort. This indicates that
  // we want to divide io, into a bidirectional token stream with the input
  // token corresponding to all of the inputs of this BlackBox, and the output token consisting of
  // all of the outputs from the BlackBox
  val bridgeIO = HostPort(io)

  // And then implement the constructorArg member
  val constructorArg = Some(GraphicsBridgeKey())

  // Finally, and this is critical, emit the Bridge Annotations -- without
  // this, this BlackBox would appear like any other BlackBox to Golden Gate
  generateAnnotations()
}
// DOC include end: UART Bridge Target-Side Module

// DOC include start: UART Bridge Companion Object

// The bridge stub is instantiated in the Test Harness (BridgeBinders), and it gets connected to the top-level port coming out of the chip 
object GraphicsBridge {
  def apply(clock: Clock, graphicsIO: chipyard.iobinders.GraphicsPortPeripheralIO, reset: Bool)(implicit p: Parameters): GraphicsBridge = {
    val ep = Module(new GraphicsBridge())
    graphicsIO.tx <> ep.io.graphics.tx 
    ep.io.graphics.guestTransmit := graphicsIO.guestTransmit
    graphicsIO.rx <> ep.io.graphics.rx
    graphicsIO.hostTransmit := ep.io.graphics.hostTransmit
    ep.io.clock := clock
    ep.io.reset := reset
    ep
  }
}
// DOC include end: UART Bridge Companion Object
