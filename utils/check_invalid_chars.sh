#!/usr/bin/env bash
grep -v '>' $1 | egrep -i "r|y|k|m|s|w|b|d|h|v|n"
