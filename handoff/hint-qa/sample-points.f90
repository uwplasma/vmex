program sample_points
  use kind_spec
  use file_name_mod, only: mag_file
  implicit none

  integer :: count, index
  real(DP) :: radius, phi, z, b_r, b_phi, b_z, b_magnitude

  call get_command_argument(1, mag_file)
  if (len_trim(mag_file) == 0) error stop "supply a native field file"

  call read_eq_field
  call vsetup
  call magset

  read(*, *) count
  if (count < 1) error stop "point count must be positive"
  open(71, file="native-points.csv", status="replace", action="write")
  do index = 1, count
    read(*, *) radius, phi, z
    call mgval1(radius, phi, z, b_r, b_phi, b_z, b_magnitude)
    write(71, '(3(ES25.16,:,","))') b_r, b_phi, b_z
  end do
  close(71)

  call free_mem
end program sample_points
