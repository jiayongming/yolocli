#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Commands module"""

# 导出所有命令模块
from . import model, data, train, detect, interactive, quick, predict

__all__ = ['model', 'data', 'train', 'detect', 'interactive', 'quick', 'predict']
