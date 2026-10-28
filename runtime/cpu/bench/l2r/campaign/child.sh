#!/usr/bin/env bash
exec 9>/tmp/g4c/l2r/t.fifo; sleep 1; echo enable >&9; sleep 1; echo disable >&9; sleep 0.5
