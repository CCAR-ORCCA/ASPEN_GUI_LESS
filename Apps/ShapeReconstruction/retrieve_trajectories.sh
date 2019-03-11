rsync -rav -e ssh --include '*/' --include='*_sparse_orbit*.txt' --include='*_sparse_time*.txt' --exclude='*' bebe0705@fortuna.colorado.edu:/orc_raid/bebe0705/ShapeReconstruction/output/  output/.
