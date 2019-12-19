// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "PSROIAlign.h"
#include "PSROIPool.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("ps_roi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
  m.def("ps_roi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
  m.def("ps_roi_pool_forward", &PSROIPool_forward, "PSROIPool_forward");
  m.def("ps_roi_pool_backward", &PSROIPool_backward, "PSROIPool_backward");
}
