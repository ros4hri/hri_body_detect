^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_body_detect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.1.1 (2024-08-06)
------------------
* Merge branch 'fix-monotically-increasing' into 'humble-devel'
* fix to make sure that the timestamp in ms 
  passed to mediapipe is monotocally increasing
* Contributors: ferrangebelli, lorenzoferrini

3.1.0 (2024-08-01)
------------------
* fixing wrong publishing of the body urdf
  we were both publishing the body urdf on
  /humans/bodie/<body_id>/urdf (right) and /robot_description (
  very very wrong, criminal stuff, death penalty in some countries).
  This was due to the fact that the robot_state_publisher publishes
  the input urdf on /robot_description and we (pluralis maiestatis,
  the fault is on the same person writing this long commit message)
  did not think about it.
* Contributors: lorenzoferrini

3.0.1 (2024-07-25)
------------------
* properly killing the robot_state_publishers
  spawned for the detected bodies
* handling tracking errors
* Contributors: ferrangebelli, lorenzoferrini

3.0.0 (2024-07-18)
------------------
* un-exposed the diagnostic rate
* added launch-pal dependency
* pap7 conformant
* Implement video multibody from mediapipe, with BoTSORT tracker
* Initial commit from hri_fullbody d04553ff3f1d11b1128d67abc9035e4953a6c863
* Contributors: Luka Juricic, ferrangebelli, lorenzoferrini
