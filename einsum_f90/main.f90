program main

    use einsum_module, only:  einsum
    use blas_module, only: kgemm

    implicit none

    integer, parameter :: nu = 20, no = 6
    integer :: a, b, c, d, i, j, k, l, m, n, e, f, ct_test, num_test
    real :: Voovv(no,no,nu,nu), T2(nu,nu,no,no), Vvoov(nu,no,no,nu), Vvvvv(nu,nu,nu,nu), Voooo(no,no,no,no),&
            T1(nu,no), Vvooo(nu,no,no,no), Fov(no,nu), Foo(no,no), Fvv(nu,nu), T3(nu,nu,nu,no,no,no)
    real, allocatable :: Z(:,:,:,:), Z2(:,:,:,:), Q(:,:), Q2(:,:), W(:,:,:,:,:,:), W2(:,:,:,:,:,:)
    real :: xsum

    call get_matrices(no,nu,Fov,Foo,Fvv,Voovv,Vvoov,Vvvvv,Voooo,Vvooo,T3,T2,T1)

    ct_test = 0
    num_test = 0

    print*,'++++++++++++++++TEST 1: Z(abef) = 0.5*V(mnef)T(abmn)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,nu,nu),Z2(nu,nu,nu,nu))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do e = 1,nu
                do f = 1,nu
                    Z(a,b,e,f) = 0.0
                    do m = 1,no
                        do n = 1,no
                            Z(a,b,e,f) = Z(a,b,e,f) + &
                            0.5*Voovv(m,n,e,f)*T2(a,b,m,n)
                        end do
                    end do 
                    xsum = xsum + Z(a,b,e,f)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mnef,abmn->abef',0.5*Voovv,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do e = 1,nu
                do f = 1,nu
                    xsum = xsum + Z(a,b,e,f) - Z2(a,b,e,f)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 2: Z(abij) = V(amie)T(bejm)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,no,no),Z2(nu,nu,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(a,b,i,j) = 0.0
                    do m = 1,no
                        do e = 1,nu
                            Z(a,b,i,j) = Z(a,b,i,j) + &
                            Vvoov(a,m,i,e)*T2(b,e,j,m)
                        end do
                    end do 
                    xsum = xsum + Z(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('amie,bejm->abij',Vvoov,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,b,i,j) - Z2(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 3: Z(abij) = 0.5*V(abef)T(efij)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,no,no),Z2(nu,nu,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(a,b,i,j) = 0.0
                    do e = 1,nu
                        do f = 1,nu
                            Z(a,b,i,j) = Z(a,b,i,j) + &
                            0.5*Vvvvv(a,b,e,f)*T2(e,f,i,j)
                        end do
                    end do 
                    xsum = xsum + Z(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('abfe,feij->abij',0.5*Vvvvv,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,b,i,j) - Z2(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 4: Z(amie) = V(mnef)T(afin)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,no,no,nu),Z2(nu,no,no,nu))
    xsum = 0.0
    do a = 1,nu
        do m = 1,no
            do i = 1,no
                do e = 1,nu
                    Z(a,m,i,e) = 0.0
                    do f = 1,nu
                        do n = 1,no
                            Z(a,m,i,e) = Z(a,m,i,e) + &
                            Voovv(m,n,e,f)*T2(a,f,i,n)
                        end do
                    end do 
                    xsum = xsum + Z(a,m,i,e)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mnef,afin->amie',Voovv,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do m = 1,no
            do i = 1,no
                do e = 1,nu
                    xsum = xsum + Z(a,m,i,e) - Z2(a,m,i,e)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 5: Z(bija) = 0.5*V(mnij)T(abmn)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,no,no,nu),Z2(nu,no,no,nu))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(b,i,j,a) = 0.0
                    do m = 1,no
                        do n = 1,no
                            Z(b,i,j,a) = Z(b,i,j,a) + &
                            0.5*Voooo(m,n,i,j)*T2(a,b,m,n)
                        end do
                    end do 
                    xsum = xsum + Z(b,i,j,a)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mnij,abmn->bija',0.5*Voooo,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(b,i,j,a) - Z2(b,i,j,a)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 6: Z(abij) = -V(amij)T(bm)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,no,no),Z2(nu,nu,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(a,b,i,j) = 0.0
                    do m = 1,no
                        Z(a,b,i,j) = Z(a,b,i,j) - &
                        Vvooo(a,m,i,j)*T1(b,m)
                    end do 
                    xsum = xsum + Z(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('amij,bm->abij',-Vvooo,T1,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,b,i,j) - Z2(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 7: Z(ai) = V(amie)T(em)++++++++++++++++'
    num_test = num_test + 1
    allocate(Q(nu,no),Q2(nu,no))
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
            Q(a,i) = 0.0
            do e = 1,nu 
                do m = 1,no 
                    Q(a,i) = Q(a,i) + &
                        Vvoov(a,m,i,e)*T1(e,m)
                end do 
            end do 
            xsum = xsum + Q(a,i)
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('amie,em->ai',Vvoov,T1,Q2)
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
                xsum = xsum + Q(a,i) - Q2(a,i)
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Q,Q2)

    print*,'++++++++++++++++TEST 8: Z(ai) = F(mi)T(am)++++++++++++++++'
    num_test = num_test + 1
    allocate(Q(nu,no),Q2(nu,no))
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
            Q(a,i) = 0.0
            do m = 1,no 
                Q(a,i) = Q(a,i) + &
                    Fov(m,i)*T1(a,m)
            end do 
            xsum = xsum + Q(a,i)
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mi,am->ai',Fov,T1,Q2)
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
                xsum = xsum + Q(a,i) - Q2(a,i)
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Q,Q2)

    print*,'++++++++++++++++TEST 9: Z(amef) = V(nmef)T(an)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,no,nu,nu),Z2(nu,no,nu,nu))
    xsum = 0.0
    do a = 1,nu
        do m = 1,no
            do e = 1,nu
                do f = 1,nu
                    Z(a,m,e,f) = 0.0
                    do n = 1,no
                        Z(a,m,e,f) = Z(a,m,e,f) - &
                        Voovv(n,m,e,f)*T1(a,n)
                    end do 
                    xsum = xsum + Z(a,m,e,f)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('nmef,an->amef',-Voovv,T1,Z2)
    xsum = 0.0
    do a = 1,nu
        do m = 1,no
            do e = 1,nu
                do f = 1,nu
                    xsum = xsum + Z(a,m,e,f) - Z2(a,m,e,f)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 10: Z(amij) = -0.5*V(mnef)T(aefijn)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,no,no,no),Z2(nu,no,no,no))
    xsum = 0.0
    do a = 1,nu
        do m = 1,no
            do i = 1,no
                do j = 1,no
                    Z(a,m,i,j) = 0.0
                    do e = 1,nu
                        do f = 1,nu 
                            do n = 1,no
                                Z(a,m,i,j) = Z(a,m,i,j) - &
                                0.5*Voovv(m,n,e,f)*T3(a,e,f,i,j,n)
                            end do 
                        end do
                    end do 
                    xsum = xsum + Z(a,m,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mnef,aefijn->amij',-0.5*Voovv,T3,Z2)
    xsum = 0.0
    do a = 1,nu
        do m= 1,no
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,m,i,j) - Z2(a,m,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 11: Z(abij) = F(me)T(abeijm)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,no,no),Z2(nu,nu,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(a,b,i,j) = 0.0
                    do e = 1,nu
                        do m = 1,no
                            Z(a,b,i,j) = Z(a,b,i,j) + &
                            Fov(m,e) * T3(a,b,e,i,j,m)
                        end do
                    end do 
                    xsum = xsum + Z(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('me,abeijm->abij',Fov,T3,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,b,i,j) - Z2(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)

    print*,'++++++++++++++++TEST 12: Z(abcijk) = V(amie)T(ebcmjk)++++++++++++++++'
    num_test = num_test + 1
    allocate(W(nu,nu,nu,no,no,no),W2(nu,nu,nu,no,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no
                            W(a,b,c,i,j,k) = 0.0
                            do m = 1,no 
                                do e = 1,nu 
                                    W(a,b,c,i,j,k) = W(a,b,c,i,j,k) + &
                                    Vvoov(a,m,i,e) * T3(e,b,c,m,j,k)
                                end do 
                            end do 
                            xsum = xsum + W(a,b,c,i,j,k)
                        end do
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('amie,ebcmjk->abcijk',Vvoov,T3,W2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no 
                            xsum = xsum + W(a,b,c,i,j,k) - W2(a,b,c,i,j,k)
                        end do 
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(W,W2)

    print*,'++++++++++++++++TEST 13: Z(abcijk) = 0.5*V(abef)T(efcijk)++++++++++++++++'
    num_test = num_test + 1
    allocate(W(nu,nu,nu,no,no,no),W2(nu,nu,nu,no,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no
                            W(a,b,c,i,j,k) = 0.0
                            do f = 1,nu
                                do e = 1,nu 
                                    W(a,b,c,i,j,k) = W(a,b,c,i,j,k) + &
                                    0.5*Vvvvv(a,b,e,f) * T3(e,f,c,i,j,k)
                                end do 
                            end do 
                            xsum = xsum + W(a,b,c,i,j,k)
                        end do
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('abef,efcijk->abcijk',0.5*Vvvvv,T3,W2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no 
                            xsum = xsum + W(a,b,c,i,j,k) - W2(a,b,c,i,j,k)
                        end do 
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(W,W2)

    print*,'++++++++++++++++TEST 14: Z(abcijk) = 0.5*V(mnij)T(abcmnk)++++++++++++++++'
    num_test = num_test + 1
    allocate(W(nu,nu,nu,no,no,no),W2(nu,nu,nu,no,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no
                            W(a,b,c,i,j,k) = 0.0
                            do m = 1,no 
                                do n = 1,no
                                    W(a,b,c,i,j,k) = W(a,b,c,i,j,k) + &
                                    0.5*Voooo(m,n,i,j) * T3(a,b,c,m,n,k)
                                end do 
                            end do 
                            xsum = xsum + W(a,b,c,i,j,k)
                        end do
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mnij,abcmnk->abcijk',0.5*Voooo,T3,W2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no 
                            xsum = xsum + W(a,b,c,i,j,k) - W2(a,b,c,i,j,k)
                        end do 
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(W,W2)

    print*,'++++++++++++++++TEST 15: Z(abcijk) = F(ae)T(ebcijk)++++++++++++++++'
    num_test = num_test + 1
    allocate(W(nu,nu,nu,no,no,no),W2(nu,nu,nu,no,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no
                            W(a,b,c,i,j,k) = 0.0
                            do e = 1,nu
                                W(a,b,c,i,j,k) = W(a,b,c,i,j,k) + &
                                Fvv(a,e) * T3(e,b,c,i,j,k)
                            end do 
                            xsum = xsum + W(a,b,c,i,j,k)
                        end do
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('ae,ebcijk->abcijk',Fvv,T3,W2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do c = 1,nu 
                do i = 1,no
                    do j = 1,no
                        do k = 1,no 
                            xsum = xsum + W(a,b,c,i,j,k) - W2(a,b,c,i,j,k)
                        end do 
                    end do
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(W,W2)

    print*,'++++++++++++++++TEST 16: Z(ai) = F(me)T(aeim)++++++++++++++++'
    num_test = num_test + 1
    allocate(Q(nu,no),Q2(nu,no))
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
            Q(a,i) = 0.0
            do m = 1,no 
                do e = 1,nu
                    Q(a,i) = Q(a,i) + &
                        Fov(m,e)*T2(a,e,i,m)
                end do 
            end do 
            xsum = xsum + Q(a,i)
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('me,aeim->ai',Fov,T2,Q2)
    xsum = 0.0
    do a = 1,nu
        do i = 1,no
                xsum = xsum + Q(a,i) - Q2(a,i)
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Q,Q2)

    print*,'++++++++++++++++TEST 17: Z(abij) = -F(mi)T(abmj)++++++++++++++++'
    num_test = num_test + 1
    allocate(Z(nu,nu,no,no),Z2(nu,nu,no,no))
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    Z(a,b,i,j) = 0.0
                    do m = 1,no
                        Z(a,b,i,j) = Z(a,b,i,j) - &
                        Foo(m,i)*T2(a,b,m,j)
                    end do 
                    xsum = xsum + Z(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'LOOP contraction = ',xsum

    call einsum('mi,abmj->abij',-Foo,T2,Z2)
    xsum = 0.0
    do a = 1,nu
        do b = 1,nu
            do i = 1,no
                do j = 1,no
                    xsum = xsum + Z(a,b,i,j) - Z2(a,b,i,j)
                end do 
            end do 
        end do 
    end do 
    print*,'EINSUM contraction error = ',xsum
    if (xsum == 0.0) then
        print*,'PASSED'
        ct_test = ct_test + 1
    else 
        print*,'FAILED' 
    end if 
    deallocate(Z,Z2)


    print*,'SUCCESSFULLY PASSED ',ct_test,'TESTS OUT OF ',num_test

    contains 

        subroutine get_matrices(no,nu,Fov,Foo,Fvv,Voovv,Vvoov,Vvvvv,Voooo,Vvooo,T3,T2,T1)

            integer, intent(in) :: no, nu
            real, intent(out) :: Voovv(no,no,nu,nu), Vvoov(nu,no,no,nu), Vvvvv(nu,nu,nu,nu), Fov(no,nu), &
                                 Foo(no,no), Fvv(nu,nu), Vvooo(nu,no,no,no),Voooo(no,no,no,no), &
                                 T3(nu,nu,nu,no,no,no), T2(nu,nu,no,no), T1(nu,no)
            integer :: a, b, i, j, m, e, f, n
            real :: r, xsum, ct

            print*,'Creating test matrices...'

            ct = 1.0
            xsum = 0.0
            do i = 1,no
                do a = 1,nu     
                    Fov(i,a) = ct 
                    ct = ct + 1.0 
                    xsum = xsum + Fov(i,a) 
                end do 
            end do 
            print*,'|F(ov)| = ',xsum

            ct = 16.0
            xsum = 0.0
            do i = 1,no
                do j = 1,no     
                    Foo(i,j) = ct 
                    ct = ct + 1.0 
                    xsum = xsum + Foo(i,j) 
                end do 
            end do 
            print*,'|F(oo)| = ',xsum

            ct = 22.0
            xsum = 0.0
            do a = 1,nu
                do b = 1,nu     
                    Fvv(a,b) = ct 
                    ct = ct + 1.0 
                    xsum = xsum + Fvv(a,b) 
                end do 
            end do 
            print*,'|F(vv)| = ',xsum

            xsum = 0.0
            ct = 1.0
            do a = 1,nu
                do b = 1,nu
                    do i = 1,no
                        do j = 1,no

                            Voovv(j,i,b,a) = ct 
                            ct = ct + 1.0

                            xsum = xsum + Voovv(j,i,b,a)

                        end do
                    end do
                end do
            end do
            print*,'|V(oovv)| = ',xsum

            xsum = 0.0
            ct = 1.0;
            do a = 1,nu
                do m = 1,no
                    do i = 1,no
                        do e = 1,nu
                            Vvoov(e,i,m,a) = ct 
                            ct = ct + 1.0

                            xsum = xsum + Vvoov(e,i,m,a)

                        end do
                    end do
                end do
            end do
            print*,'|V(voov)| = ',xsum

            xsum = 0.0
            ct = 1.0
            do a = 1,nu
                do b = 1,nu
                    do e = 1,nu 
                        do f = 1,nu 
                            Vvvvv(f,e,b,a) = ct 
                            ct = ct + 1.0

                            xsum = xsum + Vvvvv(f,e,b,a)
  
                        end do 
                    end do 
                end do 
            end do
            print*,'|V(vvvv)| = ',xsum

            xsum = 0.0
            ct = 1.0
            do m = 1,no
                do n = 1,no
                    do i = 1,no
                        do j = 1,no 
                            Voooo(j,i,n,m) = ct 
                            ct = ct + 1.0

                            xsum = xsum + Voooo(j,i,n,m)
  
                        end do 
                    end do 
                end do 
            end do
            print*,'|V(oooo)| = ',xsum

            xsum = 0.0
            ct = 10.0
            do a = 1,nu
                do m = 1,no
                    do i = 1,no
                        do j = 1,no 
                            Vvooo(a,m,i,j) = ct 
                            ct = ct + 1.0

                            xsum = xsum + Vvooo(a,m,i,j)
  
                        end do 
                    end do 
                end do 
            end do
            print*,'|V(vooo)| = ',xsum

            xsum = 0.0
            ct = 0.5
            do i = 1,no
                do j = 1,no
                    do k = 1,no 
                        do a = 1,nu
                            do b = 1,nu
                                do c = 1,nu
                                    T3(c,b,a,k,j,i) = ct
                                    ct = ct + 1.0

                                    xsum = xsum + T3(c,b,a,k,j,i)
                                end do 
                            end do
                        end do
                    end do 
                end do
            end do
            print*,'|T3| = ',xsum

            xsum = 0.0
            ct = 1.0
            do i = 1,no
                do j = 1,no
                    do a = 1,nu
                        do b = 1,nu

                            T2(b,a,j,i) = ct
                            ct = ct + 1.0

                            xsum = xsum + T2(b,a,j,i)

                        end do
                    end do
                end do
            end do
            print*,'|T2| = ',xsum

            xsum = 0.0
            ct = 1.0
            do i = 1,no
                do a = 1,nu 
                    T1(a,i) = ct 
                    ct = ct + 1.0 

                    xsum = xsum + T1(a,i)

                end do 
            end do 
            print*,'|T1| = ',xsum

        end subroutine get_matrices

end program main





    !!! order in reshape(SOURCE,SHAPE,ORDER) works very strangely...
    ! ORDER is an array defined such that if the RESHAPED array is taken with 
    ! indices given by ORDER, it will return SOURCE

    ! i know what the problem is
    ! idx = [icontr, iuncontr] is identified using the SOURCE ordering \
    ! but reshape requires ORDER to be in terms of the RESHAPED ordering
    ! e.g. consider this
    ! bejm -> mebj
    ! b goes to position 3 so order(1) = 3
    ! e goes to position 2 so order(2) = 2
    ! j goes to position 4 so order(3) = 4
    ! m goes to position 1 so order(4) = 1
    ! whereas looking at the contraction string, we identify 
    ! icontr = [4,2] and iuncotr = [1,3] hence we had 
    ! ORDER = [4,2,1,3]

    ! if we have ORDER_X = [4,2,1,3], we must
    ! this defines mebj
    ! ORDER_Y(k) = q where SOURCE(ORDER_X(q)) = SOURCE(k)
    ! e.g. k = 1 -> SOURCE(ORDER_X(q)) = SOURCE(1) -> ORDER_X(q) = 1
    !               ORDER_X equals 1 at position 3 so q = 3

    ! allocate(Z(nu,nu,no,no),Vp(nu,no,no,nu),Tp(no,nu,nu,no))
    ! Vp = reshape(Vvoov,shape=(/nu,no,no,nu/),order=(/1,3,2,4/)) ! amie -> aime
    ! Tp = reshape(T,shape=(/no,nu,nu,no/),order=(/3,2,4,1/)) ! bejm -> mebj
    ! ! 4,2,1,3
    ! ! do i = 1,2 
    ! !     do j = 1,3   
    ! !         do k = 1,3
    ! !             do l = 1,2
    ! !                 print*,'B(',i,j,k,l,') = ',Tp(i,j,k,l)
    ! !             end do 
    ! !         end do 
    ! !     end do 
    ! ! end do
    ! allocate(V2(nu*no,nu*no),T2(nu*no,nu*no),Z2(nu*no,nu*no))
    ! V2 = reshape(Vp,(/nu*no,nu*no/))
    ! T2 = reshape(Tp,(/nu*no,nu*no/))
    ! Z2 = kgemm(V2,T2)
    ! ! xsum = 0.0
    ! ! do i = 1,nu*no
    ! !     do j = 1,nu*no 
    ! !         xsum = xsum + Z2(i,j) 
    ! !     end do 
    ! ! end do 
    ! ! print*,xsum 
    ! Z = reshape(Z2,(/nu,no,nu,no/)) ! aibj 
    ! Z = reshape(Z,(/nu,nu,no,no/),order=(/1,3,2,4/))
    ! xsum = 0.0
    ! do a = 1,nu
    !     do b = 1,nu
    !         do i = 1,no
    !             do j = 1,no
    !                 xsum = xsum + Z(a,b,i,j)
    !             end do 
    !         end do 
    !     end do 
    ! end do 
    ! print*,'BLAS contraction = ',xsum
    ! deallocate(Z)