**** Aircraft Maintenance Data for Stochastic Aircraft Maintenance Check Scheduling Optimization.xlsx ****

Authors: Q. Deng, B.F. Santos
Air Transport and Operations, Faculty of Aerospace Engineering, 
Delft University of Technology

Corresponding Author: Q. Deng
Contact Information: Q.Deng@tudelft.nl



** General Introduction **
This dataset contains aircraft maintenance data collected for the AIRMES Project (www.airmes-project.eu). 



** Purpose **
The maintenance data sets are the inputs for stochastic aircraft A- and C-check scheduling optimization of 2017-2020 and 2019-2022.




** Description**



-- C_INITIAL --
FLEET:  Aircraft Type
A/C TAIL:  Aircraft tail number
DY-C:  The elapsed calendar days since the previous C-check
FH-C:  The cumulative FH since the previous C-check
FC-C:  The cumulative FC since the previous C-check
C-CI-DY:  C-check interval with respect to DY
C-CI-FH:  C-check interval with respect to FH
C-CI-FC:  C-check interval with respect to FC
C-TOL-DY:  The tolerance allowed for C-check with respect to DY
C-TOL-FH:  The tolerance allowed for C-check with respect to FH
C-TOL-FC:  The tolerance allowed for C-check with respect to FC
C-TOLU-DY:  Tolerance used in the previous C-check with respect to DY
C-TOLU-FH:  Tolerance used in the previous C-check with respect to FH
C-TOLU-FC:  Tolerance used in the previous C-check with respect to FC



-- A_INITIAL --
FLEET:  Aircraft Type
A/C TAIL:  Aircraft tail number
DY-A:  The elapsed calendar days since the previous A-check
FH-A:  The cumulative FH since the previous A-check
FC-A:  The cumulative FC since the previous A-check
A-CI-DY:  A-check interval with respect to DY
A-CI-FH:  A-check interval with respect to FH
A-CI-FC:  A-check interval with respect to FC
A-TOL-DY:  The tolerance allowed for A-check with respect to DY
A-TOL-FH:  The tolerance allowed for A-check with respect to FH
A-TOL-FC:  The tolerance allowed for A-check with respect to FC
A-TOLU-DY:  Tolerance used in the previous A-check with respect to DY
A-TOLU-FH:  Tolerance used in the previous A-check with respect to FH
A-TOLU-FC:  Tolerance used in the previous A-check with respect to FC



-- DFH --		
Average daily FH of each month in a year


-- DFC -- 		
Average daily FC of each month in a year


-- ADDITIONAL --
BEGIN YEAR:  Begin year of planning
TOTAL YEARS:  Total years in the planning horizon
BEGIN DAY:  The first day of planning
M COST:  Penalty for grounding aircraft when there is no available A-/C-check slot
C COST:  Penalty for having one additional A-/C-check slot
MAX C CHECK:  Hangar capacity with respect to C-check
MAX A CHECK:  Hangar capacity with respect to A-check
START DAY INTERVAL:  The minimum days between the start dates of two C-checks (due to maintenance resource preparation)


-- C_PEAK --
YEAR:  Year indicator of the commercial peak season
PEAK BEGIN:  Start of the commercial peak season
PEAK END:  End of the commercial peak season


-- C_NOT_ALLOWED --		
BEGIN:  Start of other small periods that C-checks are not allowed to perform
END:  End of other small periods that C-checks are not allowed to perform


-- PUBLIC HOLIDAY --
The public holidays when maintenance work is interrupted


-- A_NOT_ALLOWED -- 
Weekdays or weekends when A-check should be avoided


-- STOCHASTIC --
Historical C-check elapsed time


-- FH_STD --
The standard deviation of aircraft daily FH from Jan to Dec


-- FH_MAX --
Max aircraft daily FH from Jan to Dec



-- FH_MIN --
Min aircraft daily FH from Jan to Dec
