save all


FOMx = exactSol_x
FOMy = exactSol_y
FOMz = exactSol_z

GROMx = GIntRom1_x
GROMy = GIntRom1_y
GROMz = GIntRom1_z

GOpInfROMx = GOpRom1_x
GOpInfROMy = GOpRom1_y
GOpInfROMz = GOpRom1_z

CHOpROMx = CHOpRom2_x
CHOpROMy = CHOpRom2_y
CHOpROMz = CHOpRom2_z

NCHOpROMx = NCHOpRom3_x
NCHOpROMy = NCHOpRom3_y
NCHOpROMz = NCHOpRom3_z

AbsErrGROM_x = ABS(FOMx-GROMx) 
AbsErrGROM_y = ABS(FOMy-GROMy) 
AbsErrGROM_z = ABS(FOMz-GROMz) 

AbsErrGOpInfROM_x = ABS(FOMx-GOpInfROMx) 
AbsErrGOpInfROM_y = ABS(FOMy-GOpInfROMy) 
AbsErrGOpInfROM_z = ABS(FOMz-GOpInfROMz) 

AbsErrCHOpROM_x = ABS(FOMx-CHOpROMx) 
AbsErrCHOpROM_y = ABS(FOMy-CHOpROMy) 
AbsErrCHOpROM_z = ABS(FOMz-CHOpROMz) 

AbsErrNCHOpROM_x = ABS(FOMx-NCHOpROMx) 
AbsErrNCHOpROM_y = ABS(FOMy-NCHOpROMy) 
AbsErrNCHOpROM_z = ABS(FOMz-NCHOpROMz) 

exit
