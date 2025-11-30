import os
import sys
from typing import Optional

# 国内稳定镜像源列表（按优先级排序，标注优缺点）
HF_MIRRORS = {
    'hf-mirror': {
        'endpoint': 'https://hf-mirror.com',
        'description': '第三方稳定镜像 - 无 SSL 问题，速度快，覆盖广（备用）'
    },
    'tsinghua': {
        'endpoint': 'https://hf-mirror.tuna.tsinghua.edu.cn',
        'description': '清华大学镜像 - 学术网环境优先，稳定可靠'
    },
    'ustc': {
        'endpoint': 'https://hf-mirror.ustc.edu.cn',
        'description': '中国科学技术大学镜像 - 华东地区速度优'
    },
    'openxlab': {
        'endpoint': 'https://hub-mirror.c.302.ai',
        'description': '浦源社区镜像 - 部分网络可用，但存在 SSL 自签名证书问题'
    },
}

def setup_huggingface_mirror(
    mirror: str = 'hf-mirror',
    disable_ssl_verify: bool = False
) -> None:
    """
    设置 Hugging Face 镜像源，支持 SSL 证书验证开关

    Args:
        mirror: 镜像名称，可选值：'hf-mirror'（推荐 - 最稳定）、'tsinghua'、'ustc'、'openxlab'
        disable_ssl_verify: 是否禁用 SSL 证书验证（仅 openxlab 镜像需要开启，存在安全风险）
    """
    # 校验镜像名称，默认使用最稳定的 hf-mirror
    if mirror not in HF_MIRRORS:
        print(f"[WARNING] 未知镜像源: {mirror}，自动切换到最稳定的 hf-mirror")
        mirror = 'hf-mirror'
    
    # 设置镜像端点
    endpoint = HF_MIRRORS[mirror]['endpoint']
    os.environ['HF_ENDPOINT'] = endpoint
    print(f"[INFO] 已设置 HF_ENDPOINT: {endpoint}")
    print(f"[INFO] 镜像说明: {HF_MIRRORS[mirror]['description']}")

    # 处理 SSL 证书验证（仅为 openxlab 镜像提供临时解决方案）
    if disable_ssl_verify:
        if mirror != 'openxlab':
            print(f"[WARNING] 仅 openxlab 镜像需要禁用 SSL 验证，当前镜像 {mirror} 无需开启")
            return
        
        # 禁用 SSL 证书验证（警告：仅测试/内网环境使用，生产环境不建议）
        try:
            import ssl
            import urllib3
            # 全局禁用 SSL 验证
            ssl._create_default_https_context = ssl._create_unverified_context
            # 屏蔽 urllib3 的不安全请求警告
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            print("[WARNING] 已临时禁用 SSL 证书验证！存在安全风险，仅建议测试使用！")
        except ImportError:
            print("[ERROR] 禁用 SSL 验证失败，请安装依赖：pip install urllib3")

def get_mirror_endpoint(mirror: str = 'hf-mirror') -> str:
    """
    获取指定镜像的端点 URL

    Args:
        mirror: 镜像名称（可选值见 HF_MIRRORS）
    
    Returns:
        镜像端点 URL
    """
    if mirror not in HF_MIRRORS:
        print(f"[WARNING] 未知镜像源: {mirror}，返回推荐镜像 hf-mirror 的端点")
        mirror = 'hf-mirror'
    return HF_MIRRORS[mirror]['endpoint']

# 模块加载时自动设置（用户未手动设置 HF_ENDPOINT 时触发）
if 'HF_ENDPOINT' not in os.environ:
    # 默认使用 hf-mirror（第三方稳定镜像，无 DNS 解析问题）
    # 其他选项：'tsinghua'（清华）、'ustc'（科大）、'openxlab'（浦源）
    # 如需切换，在脚本中调用：setup_huggingface_mirror('tsinghua') 或 setup_huggingface_mirror('ustc')
    setup_huggingface_mirror('hf-mirror')