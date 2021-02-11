module einsum_module

    !use tensor_type, only: tensor_t
    use blas_module, only: kgemm 
    use sort_module, only: argsort_int, argsort
    use permute_module, only: permute2

    implicit none

    contains

        subroutine einsum222(str,A,B,C)

            character, intent(in) :: str(1:9)
            real, intent(in) :: A(:,:), B(:,:)
            real, intent(out) :: C(:,:)
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:), Bp(:,:), Cp(:,:)
            character :: s1(1:2), s2(1:2), s3(1:2)
            integer :: i, j, k, l, idx, ct1, ct2, ct3, id, &
                       idxA(1:2), idxB(1:2), idxC(1:2), idxC2(1:2),&
                       idxA2(1:2), idxB2(1:2), idxC3(1:2)
            integer :: shapeA(1:2), shapeB(1:2), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:2), temp2(1:2)
            real :: xsum

            s1 = str(1:2); s2 = str(4:5); s3 = str(8:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,2
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,2
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            C2 = kgemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum222

        subroutine einsum422(str,A,B,C)

            character, intent(in) :: str(1:11)
            real, intent(in) :: A(:,:,:,:), B(:,:)
            real, intent(out) :: C(:,:)
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:), Cp(:,:)
            character :: s1(1:4), s2(1:2), s3(1:2)
            integer :: i, j, k, l, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:2), idxC(1:2), idxC2(1:2),&
                       idxA2(1:4), idxB2(1:2), idxC3(1:2)
            integer :: shapeA(1:4), shapeB(1:2), shapeC(1:2), n1, n2, n3
            integer :: temp1(1:4), temp2(1:2)
            real :: xsum

            s1 = str(1:4); s2 = str(6:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            C2 = kgemm(A2,B2)

            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,2
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
    
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum422

        subroutine einsum424(str,A,B,C)

            character, intent(in) :: str(1:13)
            real, intent(in) :: A(:,:,:,:), B(:,:)
            real, intent(out) :: C(:,:,:,:)
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:), Cp(:,:,:,:)
            character :: s1(1:4), s2(1:2), s3(1:4)
            integer :: i, j, k, l, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:2), idxC(1:4), idxC2(1:4),&
                       idxA2(1:4), idxB2(1:2), idxC3(1:4)
            integer :: shapeA(1:4), shapeB(1:2), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:4), temp2(1:2)
            real :: xsum

            s1 = str(1:4); s2 = str(6:7); s3 = str(10:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,2
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2/)
            ct3 = 1
            do i = 1,2
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,2
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            C2 = kgemm(A2,B2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)

        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum424

        subroutine einsum444(str,A,B,C)

            character, intent(in) :: str(1:15)
            real, intent(in) :: A(:,:,:,:), B(:,:,:,:)
            real, intent(out) :: C(:,:,:,:)
            real, allocatable :: A2(:,:), B2(:,:), C2(:,:),&
                                 Ap(:,:,:,:), Bp(:,:,:,:), Cp(:,:,:,:)
            character :: s1(1:4), s2(1:4), s3(1:4)
            integer :: i, j, k, l, idx, ct1, ct2, ct3, id, &
                       idxA(1:4), idxB(1:4), idxC(1:4), idxC2(1:4),&
                       idxA2(1:4), idxB2(1:4), idxC3(1:4)
            integer :: shapeA(1:4), shapeB(1:4), shapeC(1:4), n1, n2, n3
            integer :: temp1(1:4), temp2(1:4)
            real :: xsum

            s1 = str(1:4); s2 = str(6:9); s3 = str(12:);
            shapeA = shape(A); shapeB = shape(B); shapeC = shape(C);
            ct1 = 1; ct2 = 1; ct3 = 1;
            n1 = 1; n2 = 1; n3 = 1;

            do i = 1,4
                idx = findloc(s2,s1(i),1)
                if (idx == 0) then ! i is an output index
                    idx = findloc(s3,s1(i),1)
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                    idxA(ct1) = i 
                    n1 = n1 * shapeA(i)
                    ct1 = ct1 + 1 
                else ! i is contracted
                    idxB(ct2) = idx 
                    ct2 = ct2 + 1
                end if
            end do

            do i = 1,4
                idx = findloc(s3,s2(i),1)
                if (idx /= 0) then ! idx is an output index
                    idxC(ct3) = idx
                    ct3 = ct3 + 1
                end if
            end do

            temp1 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4
                if (any(temp1(i) == idxA(1:ct1-1))) then
                    cycle
                else 
                    idxA(ct1+ct3-1) = i 
                    ct3 = ct3 + 1
                    n2 = n2 * shapeA(i)
                end if
            end do

            temp2 = (/1,2,3,4/)
            ct3 = 1
            do i = 1,4 
                if (any(temp2(i) == idxB(1:ct2-1))) then 
                    cycle 
                else 
                    idxB(ct2+ct3-1) = i 
                    ct3 = ct3 + 1 
                    n3 = n3 * shapeB(i)
                end if 
            end do

            shapeA = shapeA(idxA)
            shapeB = shapeB(idxB)
            shapeC = shapeC(idxC)

            allocate(Ap(shapeA(1),shapeA(2),shapeA(3),shapeA(4)))
            allocate(Bp(shapeB(1),shapeB(2),shapeB(3),shapeB(4)))
            allocate(Cp(shapeC(1),shapeC(2),shapeC(3),shapeC(4)))
            allocate(A2(n1,n2),B2(n2,n3),C2(n1,n3))

            ! find order of A
            do i = 1,4
                id = findloc(idxA,i,1)
                idxA2(i) = id
            end do
            Ap = reshape(A,shape=shapeA,order=idxA2)

            ! find order of B
            do i = 1,4
                id = findloc(idxB,i,1)
                idxB2(i) = id
            end do
            Bp = reshape(B,shape=shapeB,order=idxB2) 

            A2 = reshape(Ap,shape=(/n1,n2/))
            B2 = reshape(Bp,shape=(/n2,n3/))
            C2 = kgemm(A2,B2)

            ! Cp = reshape(C2,shape=shapeC)
            ! idxC2 = argsort_int(idxC)
            ! C = reshape(Cp,shape=shapeC(idxC2),order=idxC2)
            
            ! find order of C
            idxC2 = argsort_int(idxC)
            do i = 1,4
                id = findloc(idxC2,i,1)
                idxC3(i) = id
            end do
            C = reshape(C2,shape=shapeC(idxC2),order=idxC3)
        
        deallocate(Ap,Bp,A2,B2,C2)

        end subroutine einsum444



end module einsum_module
