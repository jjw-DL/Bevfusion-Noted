#include <torch/torch.h>

// CUDA function declarations
void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _x.size(0); // 3666928
  int c = _x.size(1); // 80
  int n_intervals = _interval_lengths.size(0); // (203770,)
  // 初始化输入Tensor数据指针
  const float* x = _x.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options =
      torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  // 定义输出Tensor
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);
  // 定义输出Tensor数据指针
  float* out = _out.data_ptr<float>();
  bev_pool(
    b, d, h, w, n, c, n_intervals, x,
    geom_feats, interval_starts, interval_lengths, out
  );
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    x_grad           : output features, FloatTensor[n, 4]
*/
at::Tensor bev_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);
  float* x_grad = _x_grad.data_ptr<float>();
  
  bev_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad,
    geom_feats, interval_starts, interval_lengths, x_grad
  );
  
  return _x_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_forward", &bev_pool_forward,
        "bev_pool_forward");
  m.def("bev_pool_backward", &bev_pool_backward,
        "bev_pool_backward");
}
