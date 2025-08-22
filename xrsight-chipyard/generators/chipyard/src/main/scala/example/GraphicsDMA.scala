package chipyard.example

import chisel3._
import chisel3.util._
import freechips.rocketchip.subsystem._
import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.prci._
import midas.targetutils.{SynthesizePrintf}

/*
Goal of DMA device is to write the data from the GraphicsTL peripheral to memory (rx_stream_data) 
and read the data from memory to GraphicsTL peripheral (tx_stream_data)
*/

class GraphicsDMA(beatBytes: Int)(implicit p: Parameters) extends ClockSinkDomain(ClockSinkParameters())(p){
  val node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "graphicsdma", sourceId = IdRange(0, 1))))))

  override lazy val module = new GraphicsDMAModuleImp(this)

  class GraphicsDMAModuleImp(outer: GraphicsDMA) extends Impl {
    val config = p(GraphicsKey).get

    val io = IO(new Bundle{
      // data to write to memory
      val rx = Flipped(Decoupled(UInt(512.W)))              

      // data read from memory
      val tx = Decoupled(UInt(512.W))                       

      // total size of transfer operation- decoupled so that we can tell peripheral when we are ready
      // backpressure is passed up to the bridge driver to not make a new request until req_size is ready
      val req_rx = Flipped(Decoupled(UInt(32.W)))   

      val req_tx = Flipped(Decoupled(UInt(32.W)))   

      val mem_ready = Output(Bool())
  
    })   

    withClockAndReset(clock, reset) {
      // fifo to buffer DMA rxdata from bridge -> target
      val rxfifo_stream = Module(new Queue(UInt(512.W), 64))

      // fifo to buffer DMA txdata from target -> bridge
      val txfifo_stream = Module(new Queue(UInt(512.W), 64)) 

      // fifo to buffer the memory write requests (host -> target)
      val reqfifo_stream = Module(new Queue(UInt(32.W), 10))

      // fifo to buffer the memory read requests (target -> host)
      val transfifo_stream = Module(new Queue(UInt(32.W), 10))

      io.tx <> txfifo_stream.io.deq   

      // can't bulk connect these because we don't want to be ready during reset 
      io.rx.ready := rxfifo_stream.io.enq.ready && !reset.asBool
      rxfifo_stream.io.enq.bits := io.rx.bits
      rxfifo_stream.io.enq.valid := io.rx.valid

      io.req_rx.ready := reqfifo_stream.io.enq.ready && !reset.asBool
      reqfifo_stream.io.enq.bits := io.req_rx.bits  
      reqfifo_stream.io.enq.valid := io.req_rx.valid 

      io.req_tx.ready := transfifo_stream.io.enq.ready && !reset.asBool
      transfifo_stream.io.enq.bits := io.req_tx.bits  
      transfifo_stream.io.enq.valid := io.req_tx.valid                                            

      // mem connects to IO of this module, to actually send/recieve TL messages
      // edge represents edge of diplomacy graph and has methods for constructing TL messages and retrieving data from them
      val (mem, edge) = outer.node.out(0)
      val addrBits = edge.bundle.addressBits
      val blockBytes = p(CacheBlockBytes)


      /*
      When we get a valid rx request, transition to the writing state, where we keep reading valid data from rxfifo
      and writing it to the memory.

      When we get a valid tx_request, transition to reading state where we keep buffering into txfifo
      */

      val s_init :: s_write :: s_read :: s_chunk :: s_resp :: Nil = Enum(5) 
      
      // state for FSM
      val state = RegInit(s_init)
      val prev_state = RegNext(state, s_init)

      // indicate if reading or writing
      val reading = RegInit(false.B)

      // for reading read responses
      // val shift_const: Int = beatBytes*8
      // val shift_amt = RegInit(0.U(log2Ceil(blockBytes).W)) 
      val shift_amt: Int = beatBytes*8

      val addr = Reg(UInt(addrBits.W))
      val bytesLeft = RegInit(0.U(32.W))

      // indicate to target when memory operation is complete 
      io.mem_ready := (state == s_init).asBool

      // buffers to hold read/write data
      val write_buffer  = RegInit(0.U(512.W))
      val read_buffer   = RegInit(0.U(512.W))

      // the data we read is valid when we came back from the response state to s_init (finished entire read) or s_read (finished 512b chunk)
      txfifo_stream.io.enq.bits   := read_buffer
      txfifo_stream.io.enq.valid  := (state === s_read || state === s_init) && (prev_state === s_resp) && reading.asBool 

      // counter to keep track of how many bytes we have read or written
      val rw_bytes = RegInit(0.U(32.W))   

      // advance the address after this chunk of bytes are written
      addr := config.base.U + rw_bytes         

      // can latch the next piece of data from the rx_fifo in peripheral when in write state
      rxfifo_stream.io.deq.ready := state === s_write              

      // if we are in s_chunk state, it means we have a valid request for the memory 
      mem.a.valid := state === s_chunk          

      // Put writes data to memory, Get reads data from memory
      mem.a.bits := Mux(reading.asBool, 
                          edge.Get(fromSource = 0.U, toAddress = addr, lgSize = log2Ceil(blockBytes).U)._2,
                          edge.Put(fromSource = 0.U, toAddress = addr, lgSize = log2Ceil(blockBytes).U, data = write_buffer)._2)

      // we are ready to accept a response from the memory system after each block write
      mem.d.ready := state === s_resp               

      // we are ready to serve a new read or write request when in s_init 
      reqfifo_stream.io.deq.ready   := state === s_init      
      transfifo_stream.io.deq.ready := state === s_init   
      
      switch(state) {
        is(s_init) {

          rw_bytes := 0.U     
          reading := false.B
          
          when(reqfifo_stream.io.deq.fire) {                // serve write request
            bytesLeft := reqfifo_stream.io.deq.bits
            
            state := s_write 
          } .elsewhen(transfifo_stream.io.deq.fire) {       // serve read request
            bytesLeft := transfifo_stream.io.deq.bits
            reading := true.B
            read_buffer := 0.U(512.W)
            state := s_read 
          }

        }
        is(s_write) {

          // Once we get valid data from the peripheral rx FIFO, we proceed to writing this chunk (512b) of data 
          when(rxfifo_stream.io.deq.fire) {
            write_buffer := rxfifo_stream.io.deq.bits
            state := s_chunk
          }

        }
        is(s_read) {

          // SynthesizePrintf(printf("[DMA] Entered s_read: state: %d, reading: %d, read_buffer: %x, bytesLeft: %x, txfifo_stream.io.enq.ready: %d, txfifo_stream.io.enq.valid: %d, txfifo_stream.io.deq.ready: %d, txfifo_stream.io.deq.valid: %d \n", state, reading, read_buffer, bytesLeft, txfifo_stream.io.enq.ready, txfifo_stream.io.enq.valid, txfifo_stream.io.deq.ready, txfifo_stream.io.deq.valid))
          // When the txfifo is ready to accept the next chunk of data, we proceed to read it
          when(txfifo_stream.io.enq.ready) {
            state := s_chunk
          }

        }
        is(s_chunk) {

          // SynthesizePrintf(printf("[DMA] Entered s_read: state: %d, reading: %d, read_buffer: %x, bytesLeft: %x, txfifo_stream.io.enq.ready: %d, txfifo_stream.io.enq.valid: %d, txfifo_stream.io.deq.ready: %d, txfifo_stream.io.deq.valid: %d \n", state, reading, read_buffer, bytesLeft, txfifo_stream.io.enq.ready, txfifo_stream.io.enq.valid, txfifo_stream.io.deq.ready, txfifo_stream.io.deq.valid))
          // on the last beat of this transaction on the A channel, move to response state to check D channel
          when(edge.done(mem.a)) {
            rw_bytes := rw_bytes + blockBytes.U      
            bytesLeft := bytesLeft - blockBytes.U
            state := s_resp
          }

        }
        is(s_resp) {

          // Every time a transaction is complete (response from the memory system for this chunk), we check if we're done with the whole request or need to keep reading/writing.
          
          when(mem.d.fire) {

            when(reading.asBool) {
              // in a read, AccessAckData may take multiple beats to transfer the data for this chunk
              // stay in s_resp until all the beats for this are done, and accumulate the output data

              // SynthesizePrintf(printf("[DMA] Entered s_read: state: %d, reading: %d, read_buffer: %x, bytesLeft: %x, txfifo_stream.io.enq.ready: %d, txfifo_stream.io.enq.valid: %d, txfifo_stream.io.deq.ready: %d, txfifo_stream.io.deq.valid: %d \n", state, reading, read_buffer, bytesLeft, txfifo_stream.io.enq.ready, txfifo_stream.io.enq.valid, txfifo_stream.io.deq.ready, txfifo_stream.io.deq.valid))

              // read_buffer := (read_buffer << shift_amt) + mem.d.bits.data
              // read_buffer := (mem.d.bits.data << shift_amt) + read_buffer
              read_buffer := Cat(mem.d.bits.data, read_buffer(511, shift_amt))

              // when we have received all beats of AccessAckData, go back to s_init if that is the last chunk, or s_read if we have not read enough bytes yet 
              when (edge.done(mem.d)) {
                when (bytesLeft === 0.U) {
                  // SynthesizePrintf(printf("[DMA] all done"))
                  state := s_init
                }.otherwise {
                  state := s_read
                }
              }.otherwise {
                state := s_resp
              }

            }.otherwise {
              // in a write, AccessAck response only takes 1 beat so we can determine what to do on the next cycle
              when (bytesLeft === 0.U) {
                state := s_init
              }.otherwise {
                state := s_write
              }
              
            }

            
          }
          
        }
      }

    }
  }

}



// // DOC include start: WithInitZero
// class WithInitZero(base: BigInt, size: BigInt) extends Config((site, here, up) => {
//   case InitZeroKey => Some(InitZeroConfig(base, size))
// })
// // DOC include end: WithInitZero
