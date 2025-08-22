// See LICENSE for license details

#ifndef __GRAPHICS_H
#define __GRAPHICS_H

#include "bridges/serial_data.h"
#include "core/bridge_driver.h"
#include "core/stream_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <signal.h>
#include <string>
#include <vector>
#include <fstream>

#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <algorithm>

#include <cctype>

#include <unordered_map>

/**
 * Structure carrying the addresses of all fixed MMIO ports.
 *
 * This structure is instantiated when all bridges are populated based on
 * the target configuration.
 */
struct GRAPHICSBRIDGEMODULE_struct {
  uint64_t out_bits;
  uint64_t out_valid;
  uint64_t out_ready;
  uint64_t in_bits;
  uint64_t in_valid;
  uint64_t in_ready;
  
  uint64_t guest_transmit;
  uint64_t host_transmit;
};

class graphics_t;

class graphics_handler {
  public:
    virtual ~graphics_handler() = default;

    graphics_handler(graphics_t* driver);
  
    std::optional<uint32_t> get();
    void put(uint32_t data);
    void close();

  private:
    const char* SOCKET_PATH = "/illixr-host";
    int sockfd;
    struct sockaddr_un addr;
    int BUFFER_SIZE = 1024;

    uint32_t txbuffer[50];
    uint32_t rxbuffer[50];
    
    int packet_receive_total;
    int packet_receive_count;

    uint32_t packet_send_total;
    int packet_send_count;

    graphics_t* driver;

};


class graphics_t final : public bridge_driver_t {// public streaming_bridge_driver_t {
public:
  /// The identifier for the bridge type used for casts.
  static char KIND;

  graphics_t(simif_t &simif,
        const GRAPHICSBRIDGEMODULE_struct &mmio_addrs,
        int graphicsno,
        const std::vector<std::string> &args);

  ~graphics_t() override;

  void tick() override;
  void finish() override;

  void pause_target();
  void resume_target();


private:
  const GRAPHICSBRIDGEMODULE_struct mmio_addrs;
  std::unique_ptr<graphics_handler> handler;

  serial_data_t<uint32_t> data;

  void send();
  void recv();

  bool target_paused;

};

#endif // __GRAPHICS_H
