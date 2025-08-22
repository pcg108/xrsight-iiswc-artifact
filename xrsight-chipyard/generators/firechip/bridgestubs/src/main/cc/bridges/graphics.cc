// See LICENSE for license details

#include "graphics.h"
#include "core/simif.h"

#include <fcntl.h>
#include <sys/stat.h>

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <inttypes.h>
#include <bit>
#include <cstdint>

#ifndef _WIN32
#include <unistd.h>

char graphics_t::KIND;

#endif


std::optional<uint32_t> graphics_handler::get() {

  /*
    graphics_handler::get() deals with reading from the ILLIXR host worker socker server and translating them into 4-byte packets that can be sent over MMIO
  */

  // if we have no current in-flight message, check if host worker server has a new message
  if (packet_send_count == packet_send_total) {

    ssize_t bytes_received_from_illixr = ::recv(sockfd, txbuffer, BUFFER_SIZE - 1, 0);
    if (bytes_received_from_illixr > 0) {

      // std::cout << "[bridge driver] message from illixr host with size: " << bytes_received_from_illixr << std::endl;
      packet_send_total = txbuffer[0]+1;
      packet_send_count = 1; // not sending the number of packets, that is for the bridge driver to count

      // target should have been paused when reading from bridge so now we unpause it
      driver->resume_target();
    } 

  }

  // send an in-flight message
  if (packet_send_count < packet_send_total) {
    // std::cout << "[bridge driver] sending: " << txbuffer[packet_send_count] << std::endl;
    return txbuffer[packet_send_count++];
  }
  

  return std::nullopt;
}

void graphics_handler::put(uint32_t data) {

  /*
    graphics_handler::put() reads packets from the bridge and accumulates them into a socket message for the ILLIXR server
  */
  

  if (packet_receive_total == 0) {

    if ((data & 0xFF000000) != 0xFF000000) {
      std::cout << "[bridge driver] Error: first packet was not a gpu-command-start: " << data << std::endl;
      return;
    }

    // this message should be a start message
    uint8_t start_stream  = (data >> 24) & 0xFF;  
    uint8_t queue_id      = (data >> 16) & 0xFF;  
    uint8_t num_packets   = (data >> 8) & 0xFF;
    uint8_t dma_read      = data & 0xFF;          

    rxbuffer[0] = (uint32_t) queue_id;
    rxbuffer[1] = (uint32_t) dma_read;

    packet_receive_total = (int) num_packets + 2;
    packet_receive_count = 2;

    // std::cout << "[bridge driver] start with queue ID: " << rxbuffer[0] << std::endl;

  } else {  

    rxbuffer[packet_receive_count++] = data;
    // std::cout << "[bridge driver] Received packet: " << data << std::endl;

    // once we get the total message assembled, we can send a socket message to ILLIXR host worker
    if (packet_receive_count == packet_receive_total) {   

      // std::cout << "[bridge driver] Received total from bridge: " << std::endl;
      // for (int i = 0; i < packet_receive_total; i++) {
      //   std::cout << "  " << rxbuffer[i] << std::endl;
      // }

      int buffer_size = sizeof(uint32_t) * packet_receive_total;
      uint8_t buffer[buffer_size];

      std::memcpy(buffer, rxbuffer, buffer_size);
      
      // send the stream message to the socket
      if (::send(sockfd, buffer, buffer_size, 0) == -1) {
        std::cout << "[bridge driver] error sending message to host: " << strerror(errno) << std::endl;
        perror("send");
      }

      // have driver pause the target simulation in the bridge module
      driver->pause_target();

      packet_receive_total = 0;

    }
  }
}



void graphics_handler::close() {
  ::close(sockfd);
}


graphics_handler::graphics_handler(graphics_t* driver) {

  /*
    Set up the socket connection to the host-gfxstream host2driver helper server
  */

  std::cout << "hello there" << std::endl;

  // Create a socket
  sockfd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
  if (sockfd == -1) {
      perror("socket");
      // std::cout << "[bridge driver] Error creating socket" << std::endl;
  }

  int flags = fcntl(sockfd, F_GETFL, 0);
  fcntl(sockfd, F_SETFL, flags | O_NONBLOCK); 


  const char* home = getenv("HOME");
  std::string socket_path = std::string(home) + std::string(SOCKET_PATH);

  // Zero out the address structure
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  // Connect to the server
  if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    perror("connect");
    ::close(sockfd);
    std::cout << "[bridge driver] Error connecting to server" << std::endl;
  } else {
    std::cout << "[bridge driver] Connected to " << socket_path << std::endl;
  }

  packet_receive_total = 0;
  packet_send_total = 0;
  packet_send_count = 0;

  this->driver = driver;

}

static std::unique_ptr<graphics_handler> create_handler(graphics_t* driver) {
  return std::make_unique<graphics_handler>(driver);
}


graphics_t::graphics_t(simif_t &simif,
                const GRAPHICSBRIDGEMODULE_struct &mmio_addrs, 
                int graphicsno, 
                const std::vector<std::string> &args)
    : bridge_driver_t(simif, &KIND), 
    mmio_addrs(mmio_addrs), 
    handler(create_handler(this)) {
      target_paused = false;
    }

graphics_t::~graphics_t() = default;

void graphics_t::send() {

  if (data.in.fire()) {                           // data.in.fire() is true if we have valid data and fifo is ready to accept
    write(mmio_addrs.in_bits, data.in.bits);      // write the data to the rxfifo.io.enq (send to the bridge)
    write(mmio_addrs.in_valid, data.in.valid);    // and mark it as valid 
  }
  if (data.out.fire()) {                          // data.out.fire() is true if valid and ready
    write(mmio_addrs.out_ready, data.out.ready);  // tell the bridge to dequeue the data from the fifo
  }
  
}

void graphics_t::recv() {
  data.in.ready = read(mmio_addrs.in_ready);    // check if the bridge ready to recieve data
  data.out.valid = read(mmio_addrs.out_valid);  // check if the data from the bridge is valid
  if (data.out.valid) {                         // if the data from the bridge is valid, read it into data.out
    data.out.bits = read(mmio_addrs.out_bits);
  }
}


void graphics_t::tick() {


  data.out.ready = true;                        // we are ready to receive data from outside
  data.in.valid = false;                        // the data we are sending to the bridge is not yet valid
  do {

    this->recv();                               // read anything coming from the bridge

    if (data.in.ready) {                        // if the bridge is ready to receive data
      if (auto bits = handler->get()) {         // get the packet incoming from handler
        data.in.bits = *bits;                   // write the bits and mark as valid
        data.in.valid = true;
      }
    }

    if (data.out.fire()) {                      // send the packet from the bridge out to handler
      handler->put(data.out.bits);
    }

    this->send();                               // send the data we wrote into data.in
    data.in.valid = false;                      // mark as invalid after sending
  } while (data.in.fire() || data.out.fire());  


}


void graphics_t::pause_target() {
  if (target_paused) {
    std::cout << "[bridge driver] ERROR: attempting to pause target even though it is already paused" << std::endl;
    return;
  } 

  // pulse this register to toggle state in bridge module
  write(mmio_addrs.host_transmit, 1);
  target_paused = true;
}

void graphics_t::resume_target() {
  if (!target_paused) {
    std::cout << "[bridge driver] ERROR: attempting to unpause target even though it is already running" << std::endl;
    return;
  }

  // pulse this register to toggle state in bridge module
  write(mmio_addrs.host_transmit, 1);
  target_paused = false;
}

void graphics_t::finish() {
  handler->close();
}




