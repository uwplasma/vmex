program magval_points
 use kind_spec
 use file_name_mod, only: mag_file
 implicit none
 integer :: n,i
 real(DP) :: r,phi,z,br,bp,bz,bb
 call get_command_argument(1,mag_file)
 call read_eq_field
 call vsetup
 call magset
 read(*,*) n
 open(71,file='native-points.csv',status='replace')
 do i=1,n
  read(*,*) r,phi,z
  call mgval1(r,phi,z,br,bp,bz,bb)
  write(71,'(3(ES25.16,:,","))') br,bp,bz
 end do
 close(71)
 call free_mem
end program
