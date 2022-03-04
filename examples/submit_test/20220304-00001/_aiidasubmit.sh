#!/bin/bash
exec > _scheduler-stdout.txt
exec 2> _scheduler-stderr.txt


 

'/home/jyu/Projects/WP-OPSP/bin/oncvpsp.x' < 'aiida.in' > 'aiida.out' 

 
