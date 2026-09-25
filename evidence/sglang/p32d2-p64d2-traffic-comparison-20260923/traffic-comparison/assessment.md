# 最终八条件访存比较

P为提示处理（Prefill）的输入词元数；D为之后实际解码前向（Decode）的次数。单请求，Qwen2.5-1.5B-Instruct，BF16（Brain Floating Point，16位），SGLang/FlashInfer；P128D2只计一次。

目标：qwen1p5b-P32D2, qwen1p5b-P64D2, qwen1p5b-P128D2, qwen1p5b-P256D2, qwen1p5b-P512D2, qwen1p5b-P128D4, qwen1p5b-P128D8, qwen1p5b-P128D16。历史P32D4/D8不计入正式分母。

NCU（NVIDIA Nsight Compute）硬件参考就绪 8/8；完整比较 2/8。尚未完成的条件不填零。

whole为连续提示及全部解码；Decode为独立采集的连续解码范围；steps为独立逐阶段范围。各范围使用三次正式实测的中位数，不混合范围。误差=100×(模拟−实测中位数)/实测中位数；零分母只报告绝对差。

L2为二级缓存（Level-2 Cache），DRAM为显存（Dynamic Random-Access Memory）；读请求扇区32字节。表内为whole范围，完整逐阶段结果及误差字节数见comparison.json。

| 条件 | 状态 | r4 L2读误差 | r4 DRAM写误差 |
|---|---|---:|---:|
| qwen1p5b-P32D2 | 完整比较；准确度未自动验收 | +0.9187% | +3.8943% |
| qwen1p5b-P64D2 | 完整比较；准确度未自动验收 | +1.4807% | +2.1027% |
| qwen1p5b-P128D2 | 待完成 | — | — |
| qwen1p5b-P256D2 | 待完成 | — | — |
| qwen1p5b-P512D2 | 待完成 | — | — |
| qwen1p5b-P128D4 | 待完成 | — | — |
| qwen1p5b-P128D8 | 待完成 | — | — |
| qwen1p5b-P128D16 | 待完成 | — | — |

配置保持冻结r4候选；未建模有限MSHR（Miss Status Holding Register，未命中状态保持寄存器）并发和真实请求时序。来源闭合不证明未采样地址及真实到达顺序均正确。无运行时间截止，不保存新的原始访存trace；保留资源预算、compact profile与失败证据。
