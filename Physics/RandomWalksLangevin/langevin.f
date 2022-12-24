      PROGRAM ECUACION_LANGEVIN
c***************************************************
c
c     Problema 4: Simulacion de la ec. de Lengevin 
c
c***************************************************
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

      PARAMETER(NUMAX = 714025)
      PARAMETER(NPART = 10000)
      EXTERNAL UNIFORMS, INVERSA
      DIMENSION uran(NUMAX)

      CALL UNIFORMS(uran, NUMAX)
      CALL dran_ini(9)

c     Datos iniciales:
      Gamma = 100000.d0
      tmax = 500.d0
      dt = 0.05d0
      v0 = 0.0d0
      h0 = 100.d0

      t_up = 0.0d0

      DO i = 1, NPART

c        Evolucion:
         vn = v0
         hn = h0
         kount = 0
         nplot = 0

         t = 0.0d0

         DO WHILE(t.LT.tmax)
       
c            nplot = nplot + 1
c            IF(nplot.GE.10) THEN
c               write(3,*) t, hn, vn
c               nplot = 0
c            ENDIF

c           por cada llamada dos pasos temporales
            CALL INVERSA(x1,x2,URAN,NUMAX)

c           primer paso
            vn = vn * (1.d0 - dt) - dt + dsqrt(2.d0 * Gamma * dt) * x1
            hn = hn + vn * dt
            t = t + dt
            
            IF(vn.GT.0.d0) kount = kount + 1
            
c           segundo paso
            vn = vn * (1.d0 - dt) - dt + dsqrt(2.d0 * Gamma * dt) * x2
            hn = hn + vn * dt
            t = t + dt

            IF(vn.GT.0.d0) kount = kount + 1

         ENDDO
         
         t_up = t_up + float(kount)

      ENDDO
         
      t_up = t_up/dfloat(NPART)
      t_total = tmax / dt

      print*, 'Magnitud de ruido: Gamma = ', Gamma
      print*, 'Fraccion de tiempo que sube: '
      print*, 't(total)/t(up) = ', t_total/t_up
      print*, 'Porcentaje de subida: ',t_up*100.d0/t_total,' %'

      STOP
      END

c=====================================================
      SUBROUTINE UNIFORMS(URAN, NUMAX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
c     Subrutina de generacion de uniformes
      DIMENSION URAN(NUMAX)

c     Semilla
      seed = 1.d0

c     Genero los uniformes y los guardo en URAN
      xn = seed
      indica = 0
      i = 0      
      
      DO WHILE (indica.eq.0)       
         i = i + 1
         xnn = 1366.d0 * xn + 150889.d0
         xnn = DMOD(xnn,714025.d0)        
         xn = xnn 
         URAN(i) = xnn / dfloat(NUMAX)
         IF(xn.eq.seed) indica = 1         
      ENDDO

      RETURN
      END
c----------------------------------------------
      SUBROUTINE INVERSA(x1,x2,URAN,NUMAX)
c     Genera dos aleatorios gaussianos mediante 
c     el metodo de la transformada inversa 
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION URAN(NUMAX)
      EXTERNAL i_dran

      pi = 4.d0 * ATAN(1.d0)

      i1 = i_dran(NUMAX)
      i2 = i_dran(NUMAX)

c     Necesito dos U(0,1)
      z1 = URAN(i1)
      z2 = URAN(i2)

c     Genero dos N(0,1)
      x1 = dsqrt(-2.0 * log(z1)) * cos(2.d0*pi * z2)
      x2 = dsqrt(-2.0 * log(z2)) * sin(2.d0*pi * z1)

      RETURN
      END
c--------------------------------------------------
      subroutine dran_ini(iseed0)

      implicit double precision(a-h,o-z)

      parameter(ip=1279)
      parameter(np=14)
      parameter(nbit=31)
      parameter(m=2**np,np1=nbit-np,nn=2**np1-1,nn1=nn+1)
      integer ix(ip)
      dimension g(0:m)
      
      data c0,c1,c2/2.515517,0.802853,0.010328/
      data d1,d2,d3/1.432788,0.189269,0.001308/
c     
      common /ixx/ ix
      common /icc/ ic     
      common /gg/ g
c     
      dseed=iseed0
      do 200 i=1,ip
         ix(i)=0
         do 200 j=0,nbit-1
            if(rand_xx(dseed).lt.0.5) ix(i)=ibset(ix(i),j)
 200     continue
         ic=0
c     
         pi=4.0d0*datan(1.0d0)
         do 1 i=m/2,m
            p=1.0-real(i+1)/(m+2)
            t=sqrt(-2.0*log(p))
            x=t-(c0+t*(c1+c2*t))/(1.0+t*(d1+t*(d2+t*d3)))
            g(i)=x
            g(m-i)=-x
 1       continue
         
         u2th=1.0-real(m+2)/m*sqrt(2.0/pi)*g(m)*exp(-g(m)*g(m)/2)
         u2th=nn1*sqrt(u2th)
         do 856 i=0,m
 856        g(i)=g(i)/u2th
            
            return
            end
C----------------------------------------      
      function i_dran(n)
      implicit double precision(a-h,o-z)

      parameter(ip=1279)
      parameter(iq=418)
      parameter(is=ip-iq)
      integer ix(ip)
      common /ixx/ ix
      common /icc/ ic
      ic=ic+1
      
      if(ic.gt.ip) ic=1
      if(ic.gt.iq) then
         ix(ic)=ieor(ix(ic),ix(ic-iq))
      else
         ix(ic)=ieor(ix(ic),ix(ic+is))
      endif
      i_ran=ix(ic)
      if (n.gt.0) i_dran=mod(i_ran,n)+1
      return
      end
C----------------------------------------------
      function dran_u()
      implicit double precision(a-h,o-z)
      parameter(ip=1279)
      parameter(iq=418)
      parameter(is=ip-iq)
      parameter (rmax=2147483647.0)
      integer ix(ip)
      common /ixx/ ix
      common /icc/ ic
      ic=ic+1
      if(ic.gt.ip) ic=1
      if(ic.gt.iq) then
         ix(ic)=ieor(ix(ic),ix(ic-iq))
      else
         ix(ic)=ieor(ix(ic),ix(ic+is))
      endif
      dran_u=real(ix(ic))/rmax
      return
      end
C----------------------------------------------      
      function rand_xx(dseed)
      double precision a,c,xm,rm,dseed,rand_xx
      parameter (xm=2.d0**32,rm=1.d0/xm,a=69069.d0,c=1.d0)
      dseed=mod(dseed*a+c,xm)
      rand_xx=dseed*rm
      return
      end
C----------------------------------------------      
