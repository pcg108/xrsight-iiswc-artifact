package chipyard.example

import sys.process._

import chisel3._
import chisel3.util._
import chisel3.experimental.{IntParam, BaseModule}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.prci._
import freechips.rocketchip.subsystem.{BaseSubsystem, PBUS, FBUS}
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{HasRegMap, RegField}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.UIntIsOneOf
import testchipip.util.{ClockedIO}


case class GraphicsParams(
  address: BigInt = 0x4000,
  width: Int = 32,
  base: BigInt = 0x88000000L,
  size: BigInt = 500000000)

case object GraphicsKey extends Field[Option[GraphicsParams]](None)

class GraphicsIO(val w: Int) extends Bundle {
  // Bridge -> Target
  val rx = new Bundle {
    val enq = Flipped(Decoupled(UInt(32.W)))
    val deq = Decoupled(UInt(32.W))
  }

  // Target -> Bridge
  val tx = new Bundle{
    val enq = Flipped(Decoupled(UInt(w.W)))
    val deq = Decoupled(UInt(w.W))
  }

}

// top-level signals of the TL module
class GraphicsTopIO extends Bundle {
  // Target recieves from bridge
  val rx = Flipped(Decoupled(UInt(32.W)))
  val hostTransmit = Input(Bool())

  // Target sends to bridge
  val tx = Decoupled(UInt(32.W))
  val guestTransmit = Output(Bool())

  // indicates if the DMA is in idle state
  // val memReady = Input(Bool())

}

trait HasGraphicsTopIO {
  def io: GraphicsTopIO
}

class GraphicsMMIOChiselModule(val w: Int) extends Module {
  val io = IO(new GraphicsIO(w))

  // fifo to buffer the txdata from target -> bridge
  val txfifo = Module(new Queue(UInt(w.W), 64))

  // connect txfifo interface to io interface
  txfifo.io.enq <> io.tx.enq // io.tx.enq is an input to this module, and we send that to enq into the fifo, whose enq interface takes bits and valid as input
  io.tx.deq <> txfifo.io.deq // io.tx.deq is an output from this module, so we drive that with the deq interface of the fifo

  // fifo to buffer the rxdata from the bridge -> target
  val rxfifo = Module(new Queue(UInt(w.W), 64))
  
  // connect rxfifo interface to io interface
  io.rx.enq.ready := rxfifo.io.enq.ready && !reset.asBool
  rxfifo.io.enq.bits := io.rx.enq.bits 
  rxfifo.io.enq.valid := io.rx.enq.valid

  rxfifo.io.deq <> io.rx.deq 

  txfifo.clock := clock
  txfifo.reset := reset

  rxfifo.clock := clock
  rxfifo.reset := reset

}

class GraphicsTL(params: GraphicsParams, beatBytes: Int)(implicit p: Parameters) extends ClockSinkDomain(ClockSinkParameters())(p) {
  val device = new SimpleDevice("GraphicsTL", Seq("ucbbar,GraphicsTL")) 
  val node = TLRegisterNode(Seq(AddressSet(params.address, 4096-1)), device, "reg/control", beatBytes=beatBytes)

  override lazy val module = new GraphicsImpl
  class GraphicsImpl extends Impl with HasGraphicsTopIO {
    val io = IO(new GraphicsTopIO)
    withClockAndReset(clock, reset) {
      
      // instantiate the MMIO Chisel Module
      val impl = Module(new GraphicsMMIOChiselModule(params.width))

      // MMIO registers for target
      val tx_data = Wire(Decoupled(UInt(params.width.W))) // data we want to send out from target
      val guestTransmit = RegInit(false.B)
      val rx_data = Wire(Decoupled(UInt(params.width.W)))
      val hostTransmit = Wire(Bool())

      val status = Wire(UInt(2.W))

      // Drive the Chip top level IO transmit with data coming from the target
      io.guestTransmit := guestTransmit
      // send the target data through the fifo to the top level
      impl.io.tx.enq <> tx_data // fifo <- target
      io.tx <> impl.io.tx.deq   // top <- fifo

      // Send the data coming from Chip top level to the target
      hostTransmit := io.hostTransmit
      io.rx <> impl.io.rx.enq   // top -> fifo
      impl.io.rx.deq <> rx_data // fifo -> target      

      // the status register should indicate to the target whether:
      // - we are ready to accept input data
      // - if there is valid output data
      status := Cat(impl.io.tx.enq.ready, impl.io.rx.deq.valid)
      // status := Cat(io.memReady, impl.io.tx.enq.ready, impl.io.rx.deq.valid)

      node.regmap(
        0x00 -> Seq(
          RegField.r(2, status)), // a read-only register capturing current status
        0x04 -> Seq(
          RegField.w(params.width, tx_data)), // register to write tx_data
        0x08 -> Seq(
          RegField.w(1, guestTransmit)), // register to write if guest is transmitting
        0x0C -> Seq(
          RegField.r(params.width, rx_data)), // register to read rx_data
        0x10 -> Seq(
          RegField.r(1, hostTransmit)) // register to read if host is transmitting
      )
    }
  }
}

/*
  This matches the GraphicsPortIO in the target-side bridge definition
  This gets connected to the bridge stub in the test harness
*/
class GraphicsPortPeripheralIO extends Bundle {

  val rx = Flipped(Decoupled(UInt(32.W)))
  val hostTransmit = Input(Bool())

  val tx = Decoupled(UInt(32.W))
  val guestTransmit = Output(Bool())

  // val rx_stream = Flipped(Decoupled(UInt(512.W)))
  // val tx_stream = Decoupled(UInt(512.W))

  // val stream_req_rx = Flipped(Decoupled(UInt(32.W)))
  // val stream_req_tx = Flipped(Decoupled(UInt(32.W)))
}

// this is a trait that instantiates the TL module defined above, and sends the ports to the toplevel of the chip
trait CanHavePeripheryGraphics { this: BaseSubsystem =>
  private val portName = "GraphicsTL"

  private val pbus = locateTLBusWrapper(PBUS)
  private val fbus = locateTLBusWrapper(FBUS)

  val graphicsio = p(GraphicsKey) match {
    case Some(params) => {
      // generate lazy module, which enables Diplomatic connections
      val graphicsTL = LazyModule(new GraphicsTL(params, pbus.beatBytes)(p))
        graphicsTL.clockNode := pbus.fixedClockNode
        pbus.coupleTo(portName) { graphicsTL.node := TLFragmenter(pbus.beatBytes, pbus.blockBytes) := _ }


      // instantiating the Graphics DMA module
      // val graphicsDMA = LazyModule(new GraphicsDMA(fbus.beatBytes)(p)) 
      // graphicsDMA.clockNode := fbus.fixedClockNode
      // fbus.coupleFrom("graphics-dma") { _ := graphicsDMA.node }
            
      // InModuleBody allows us to punch ports through to top level module (so that it can connect to bridge in this case)
      // so we connect the ports of the MMIOChiselModule we defined above to the ports that are exposed at the top
      val graphics_outer_io = InModuleBody {
        val outer_io = IO(new ClockedIO(new GraphicsPortPeripheralIO())).suggestName("GraphicsTL")
        dontTouch(outer_io)
        outer_io.clock := graphicsTL.module.clock
        outer_io.bits.rx <> graphicsTL.module.io.rx
        outer_io.bits.hostTransmit <> graphicsTL.module.io.hostTransmit
        outer_io.bits.tx <> graphicsTL.module.io.tx
        outer_io.bits.guestTransmit <> graphicsTL.module.io.guestTransmit
        
        // connect graphicsDMA IO to the outer IO 
        // outer_io.bits.rx_stream <> graphicsDMA.module.io.rx 
        // outer_io.bits.tx_stream <> graphicsDMA.module.io.tx
        // outer_io.bits.stream_req_rx <> graphicsDMA.module.io.req_rx
        // outer_io.bits.stream_req_tx <> graphicsDMA.module.io.req_tx

        // graphicsTL.module.io.memReady := graphicsDMA.module.io.mem_ready

        outer_io
      }
      Some(graphics_outer_io)
    }
    case None => None
  }
}

// added this to TargetConfigs
class WithGraphics() extends Config((site, here, up) => {
  case GraphicsKey => {
    Some(GraphicsParams())
  }
})