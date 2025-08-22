// See LICENSE for license details

package firechip.goldengateimplementations

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.Parameters

import midas.widgets._
import firesim.lib.bridgeutils._

import firechip.bridgeinterfaces._

import midas.targetutils.{FpgaDebug, SynthesizePrintf}

//Note: This file is heavily commented as it serves as a bridge walkthrough
//example in the FireSim docs

// DOC include start: UART Bridge Header
// Our UARTBridgeModule definition, note:
// 1) it takes one parameter, key, of type UARTKey --> the same case class we captured from the target-side
// 2) It accepts one implicit parameter of type Parameters
// 3) It extends BridgeModule passing the type of the HostInterface
//
// While the Scala type system will check if you parameterized BridgeModule
// correctly, the types of the constructor arugument (in this case UARTKey),
// don't match, you'll only find out later when Golden Gate attempts to generate your module.
class GraphicsBridgeModule(key: GraphicsBridgeKey)(implicit p: Parameters) 
  extends BridgeModule[HostPortIO[GraphicsBridgeTargetIO]]()(p) 
  {

  lazy val module = new BridgeModuleImp(this) {

    val hostTransmit = RegInit(false.B)
    val guestTransmit = RegInit(false.B)

    // This creates the interfaces for all of the host-side transport
    // AXI4-lite for the simulation control bus, =
    // AXI4 for DMA
    val io = IO(new WidgetIO())

    // This creates the host-side interface of your TargetIO
    // we can use the bridge->target IO definition because it is the same interface as bridge module -> bridge
    val hPort = IO(HostPort(new GraphicsBridgeTargetIO))

    // Generate some FIFOs to capture tokens...
    val rxfifo = Module(new Queue(UInt(32.W), 128))
    val txfifo = Module(new Queue(UInt(32.W), 128))

    // when the bridge driver pulses hostTransmit, toggle the target pause state
    // starts false so target starts running
    val pauseTarget = RegInit(false.B)
    when(hostTransmit.asBool) {
      pauseTarget := ~pauseTarget
    }

    val target = hPort.hBits.graphics
    // In general, your BridgeModule will not need to do work every host-cycle. In simple Bridges,
    // we can do everything in a single host-cycle -- fire captures all of the
    // conditions under which we can consume and input token and produce a new
    // output token

   //  fire when: a) we have a valid token destined for the host, b) the host is ready to accept a token, and c) the txfifo (transmit fifo) is ready to accept new data 
   // fire means when we consume a token and produce a token

    val fire = hPort.toHost.hValid &&   // We have a valid input token: toHost ~= leaving the transformed RTL
               hPort.fromHost.hReady && // We have space to enqueue a new output token
               txfifo.io.enq.ready  &&     // We have space to capture new TX data
               ~pauseTarget // target is not supposed to be paused
               
    val targetReset = fire & hPort.hBits.reset
    rxfifo.reset := reset.asBool || targetReset
    txfifo.reset := reset.asBool || targetReset

    hPort.toHost.hReady := fire
    hPort.fromHost.hValid := fire

  /*
    The bridge module connects the bridge to the host. It lives on the FPGA and can connect to the Host CPU using the bridge driver.

    the hPort has the same interface as bridge->target because it is the same direction as bridge module -> bridge

    the data to transmit from the target to the host over the bridge through this bridge driver comes from hPort.hBits.graphics.tx (the input from the bridge)
    - we write this data into txfifo.io.enq
    - the txfifo.io.deq is exposed to the bridge driver as memory mapped registers for reading

    the data to read from the host to the target over the bridge using this bridge driver comes from rxfifo.io.deq
    - rxfifo.io.enq is exposed to the bridge driver as memory mapped registers for writing
    - hPort.hBits.graphics.rx (the output to the bridge) gets written with the data from rxfifo

    Basically, we let the bridge driver read from the txfifo and write to the rxfifo. We connect those to the target through the bridge through hPort.hBits.graphics.{rx, tx}
  
    decoupled has output of ready, input of valid and bits for tx, opposite for rx
 
  */

    // connect MMIO
    txfifo.io.enq.bits := target.tx.bits
    txfifo.io.enq.valid := target.tx.valid && fire
    target.tx.ready := txfifo.io.enq.ready

    target.rx.bits := rxfifo.io.deq.bits
    target.rx.valid := rxfifo.io.deq.valid 
    rxfifo.io.deq.ready := target.rx.ready && fire

    target.hostTransmit := hostTransmit
    guestTransmit := target.guestTransmit


    // DOC include start: UART Bridge Footer
    // Exposed the head of the queue and the valid bit as a read-only registers
    // with name "out_bits" and out_valid respectively
    genROReg(txfifo.io.deq.bits, "out_bits")
    genROReg(txfifo.io.deq.valid, "out_valid")

    // Generate a writeable register, "out_ready", that when written to dequeues
    // a single element in the tx_fifo. Pulsify derives the register back to false
    // after pulseLength cycles to prevent multiple dequeues
    Pulsify(genWORegInit(txfifo.io.deq.ready, "out_ready", false.B), pulseLength = 1) // this is for the bridge driver to tell the txfifo it is ready to recieve data

    // Generate registers for the rx-side of the UART; this is eseentially the reverse of the above
    genWOReg(rxfifo.io.enq.bits, "in_bits")
    Pulsify(genWORegInit(rxfifo.io.enq.valid, "in_valid", false.B), pulseLength = 1)
    genROReg(rxfifo.io.enq.ready, "in_ready") // this is for the rxfifo to tell the bridge driver it is ready to recieve data

    // the bridge driver will read from this register to check if the guest is transmitting a message
    genROReg(guestTransmit, "guest_transmit")

    // the bridge driver will write to this register when the host is transmitting a stream message
    Pulsify(genWORegInit(hostTransmit, "host_transmit", false.B), pulseLength = 1)
    

    // This method invocation is required to wire up all of the MMIO registers to
    // the simulation control bus (AXI4-lite)
    genCRFile()
    // DOC include end: UART Bridge Footer

    override def genHeader(base: BigInt, memoryRegions: Map[String, BigInt], sb: StringBuilder): Unit = {
      genConstructor(
        base, 
        sb, 
        "graphics_t", 
        "graphics")
    }
  }
}
