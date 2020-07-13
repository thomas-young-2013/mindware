#!/usr/bin/env bash
kill -9 $(ps aux | grep "python test/*" | grep -v "grep" |tr -s " "| cut -d " " -f 2)
