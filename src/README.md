# 实现位置

生产实现已迁到 [Rust 核心](../backend/README.md) 和
[Rust 原生桌面](../desktop/README.md)。此目录不再含运算模块或桌面运行入口。

原始研究接口与模块指纹记录在 [迁移清单](../backend/migration.json)，
数值容差、冻结参考和实际验证层记录在 [验证记录](../backend/verification.md)。
旧实现可从 Git 历史获取；生产程序和测试不启动解释器。
