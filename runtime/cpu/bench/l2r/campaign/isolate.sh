#!/usr/bin/env bash
# isolate.sh apply|restore|status: reboot-free P0.2 isolation.
#   housekeeping = 2 physical cores per SNC node + SMT siblings: 0,1,32,33,64,65 + 96,97,128,129,160,161
#   inference    = physical cores 2-31,34-63,66-95 (siblings left idle)
# apply saves the original IRQ affinities, workqueue cpumask and slice AllowedCPUs to $S first;
# restore puts them back. user.slice is NOT restricted (other sessions live there).
set -u
HK=0,1,32,33,64,65,96,97,128,129,160,161
HKMASK=$(python3 -c "m=sum(1<<c for c in [0,1,32,33,64,65,96,97,128,129,160,161]); print(','.join(format((m>>(32*i))&0xffffffff,'08x') for i in reversed(range(6))))")
S=/tmp/g4c/l2r/isolate.saved
case ${1:-status} in
apply)
  if [ ! -e $S/irq.txt ]; then
    mkdir -p $S
    for d in /proc/irq/[0-9]*; do echo "$(basename $d) $(cat $d/smp_affinity_list 2>/dev/null)"; done > $S/irq.txt
    cat /sys/devices/virtual/workqueue/cpumask > $S/wq.txt
    for u in system.slice init.scope; do echo "$u $(systemctl show -p AllowedCPUs --value $u)"; done > $S/slices.txt
    systemctl is-active irqbalance > $S/irqbalance.txt 2>&1
  fi
  sudo systemctl stop irqbalance 2>/dev/null
  n=0; f=0
  for d in /proc/irq/[0-9]*; do
    if echo $HK | sudo tee $d/smp_affinity_list > /dev/null 2>&1; then n=$((n+1)); else f=$((f+1)); fi
  done
  echo "irqs moved $n, refused $f (managed/per-cpu)"
  echo $HKMASK | sudo tee /sys/devices/virtual/workqueue/cpumask > /dev/null && echo "workqueue cpumask $HKMASK"
  for u in system.slice init.scope; do sudo systemctl set-property --runtime $u AllowedCPUs=$HK && echo "$u AllowedCPUs=$HK"; done
  ;;
restore)
  [ -e $S/irq.txt ] || { echo "nothing saved"; exit 1; }
  while read irq aff; do [ -n "$aff" ] && echo $aff | sudo tee /proc/irq/$irq/smp_affinity_list > /dev/null 2>&1; done < $S/irq.txt
  sudo tee /sys/devices/virtual/workqueue/cpumask < $S/wq.txt > /dev/null
  while read u cpus; do sudo systemctl set-property --runtime $u AllowedCPUs=$cpus; done < $S/slices.txt
  grep -q '^active' $S/irqbalance.txt && sudo systemctl start irqbalance
  rm -rf $S; echo restored
  ;;
status)
  echo "workqueue: $(cat /sys/devices/virtual/workqueue/cpumask)"
  for u in system.slice init.scope user.slice; do echo "$u AllowedCPUs=$(systemctl show -p AllowedCPUs --value $u)"; done
  echo "irqbalance: $(systemctl is-active irqbalance 2>&1)"
  awk 'NR>1 && $1 ~ /^[0-9]+:/' /proc/interrupts | awk '{print $1}' | tr -d : | while read i; do cat /proc/irq/$i/effective_affinity_list 2>/dev/null; done | sort | uniq -c | sort -rn | head -5
  ;;
esac
