NR>1 { gsub(/"/,"",$9); gsub(/"/,"",$16); kn=$9;
  if($16=="MfmaUtil"){ mu[kn]+=$17; cnt[kn]++; vg[kn]=$13; ag[kn]=$14; wg[kn]=$10; lds[kn]=$11 }
  if($16=="MemUnitStalled"){ ms[kn]+=$17; msc[kn]++ } }
END{ best=""; bc=0; for(k in cnt) if(cnt[k]>bc){bc=cnt[k];best=k}
  printf "  DOMINANT(%d) MfmaUtil=%.1f MemStall=%.1f VGPR=%s AGPR=%s WG=%s LDS=%s\n    %s\n",
    bc, mu[best]/cnt[best], (msc[best]?ms[best]/msc[best]:0), vg[best],ag[best],wg[best],lds[best], best }
