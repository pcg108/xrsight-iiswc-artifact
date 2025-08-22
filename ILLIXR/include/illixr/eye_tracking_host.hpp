#pragma once

#include "data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/relative_clock.hpp"


using namespace ILLIXR;

class eye_tracking_host : public phonebook::service {
public:
    [[nodiscard]] virtual eye_position_type     get_eye_position(cv::Mat img)                              = 0;

    ~eye_tracking_host() override = default;
};
