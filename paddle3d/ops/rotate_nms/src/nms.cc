// #include "nms.h"
#include "nms_cpu.h"
PYBIND11_MODULE(rotate_nms_cpu, m)
{
    m.doc() = "non_max_suppression asd";
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<float>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &rotate_non_max_suppression_cpu<double>, py::return_value_policy::reference_internal, "bbox iou", 
          "box_corners"_a = 1, "order"_a = 2, "standup_iou"_a = 3, "thresh"_a = 4);
}