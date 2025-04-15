#include "structure.cuh"

AffinePointG1::AffinePointG1(SyncedMemory a, SyncedMemory b):x(a),y(b){}

bool AffinePointG1::is_zero(AffinePointG1 self){
    void* x_ = self.x.mutable_cpu_data();
    void* y_ = self.y.mutable_cpu_data();
    SyncedMemory one = fq::one();
    void* one_ = one.mutable_cpu_data();
    bool is_zero = true;
    for(int i = 0; i<fq::Limbs; i++){
        is_zero = is_zero && (static_cast<uint64_t*>(x_)[i] == 0) && (static_cast<uint64_t*>(y_)[i] == static_cast<uint64_t*>(one_)[i]);
    }
    return is_zero;
}

ProjectivePointG1::ProjectivePointG1(SyncedMemory a, SyncedMemory b, SyncedMemory c):x(a),y(b),z(c){}

bool ProjectivePointG1::is_zero(ProjectivePointG1 self){
    void* z_ = self.z.mutable_cpu_data();
    bool is_zero = true;
    for(int i = 0; i<fq::Limbs; i++){
        is_zero = is_zero && (reinterpret_cast<uint64_t*>(z_)[i] == 0);
    }
    return is_zero;
}


AffinePointG1 to_affine(ProjectivePointG1 input){
    if (ProjectivePointG1::is_zero(input)){
        SyncedMemory x = fq::zero();
        SyncedMemory y = fq::one();
        return AffinePointG1(x, y);
    }
    else{
        SyncedMemory one = fq::one();
        //div_mod work on cpu
        SyncedMemory zinv = div_mod(one, input.z);
        SyncedMemory zinv_squared = mul_mod(zinv, zinv);

        SyncedMemory x = mul_mod(input.x, zinv_squared);
        SyncedMemory mid1 = mul_mod(zinv_squared, zinv);
        SyncedMemory y = mul_mod(input.y, mid1);
        AffinePointG1 res = AffinePointG1(x,y);
        return res;
    }
}

ProjectivePointG1 add_assign(ProjectivePointG1 self, ProjectivePointG1 other){
    if(ProjectivePointG1::is_zero(self)){
        return ProjectivePointG1(other.x, other.y, other.z);
    }
    if(ProjectivePointG1::is_zero(other)){
        return ProjectivePointG1(self.x, self.y, self.z);
    }

    // Z1Z1 = Z1^2
    SyncedMemory z1z1 = mul_mod(self.z, self.z);

    // Z2Z2 = Z2^2
    SyncedMemory z2z2 = mul_mod(other.z, other.z);

    // U1 = X1*Z2Z2
    SyncedMemory u1 = mul_mod(self.x, z2z2);

    // U2 = X2*Z1Z1
    SyncedMemory u2 = mul_mod(other.x, z1z1);

    // S1 = Y1*Z2*Z2Z2
    SyncedMemory mid1 = mul_mod(self.y, other.z);
    SyncedMemory s1 = mul_mod(mid1, z2z2);
    
    // S2 = Y2*Z1*Z1Z1
    SyncedMemory mid2 = mul_mod(other.y, self.z);
    SyncedMemory s2 = mul_mod(mid2, z1z1);


    if (fq::is_equal(u1, u2) && fq::is_equal(s1, s2)){
        // The two points are equal, so we double.
        return double_ProjectivePointG1(self);
    }
    else{
        // H = U2-U1
        SyncedMemory h = sub_mod(u2, u1);

        // I = (2*H)^2
        SyncedMemory i = mul_mod(add_mod(h, h), add_mod(h, h));

        // J = H*I
        SyncedMemory j = mul_mod(h, i);

        // r = 2*(S2-S1)
        SyncedMemory r = add_mod(sub_mod(s2, s1), sub_mod(s2, s1));

        // V = U1*I
        SyncedMemory v = mul_mod(u1, i);

        // X3 = r^2 - J - 2*V
        SyncedMemory x = sub_mod(sub_mod(mul_mod(r, r), j), add_mod(v, v));

        // Y3 = r*(V - X3) - 2*S1*J
        SyncedMemory y = sub_mod(mul_mod(r, sub_mod(v, x)), add_mod(mul_mod(s1, j), mul_mod(s1, j)));

        // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
        SyncedMemory z = mul_mod(sub_mod(sub_mod(mul_mod(add_mod(self.z, other.z), add_mod(self.z, other.z)), z1z1), z2z2), h);
        
        return ProjectivePointG1(x,y,z);
    }
}

ProjectivePointG1 double_ProjectivePointG1(ProjectivePointG1 self){
    if (ProjectivePointG1::is_zero(self)){
        return self;
    }
    if (A == 0){
        // A = X1^2
        SyncedMemory a = mul_mod(self.x, self.x);

        // B = Y1^2
        SyncedMemory b = mul_mod(self.y, self.y);

        // C = B^2
        SyncedMemory c = mul_mod(b, b);

        // D = 2*((X1+B)^2-A-C)
        SyncedMemory mid1 = add_mod(self.x, b);
        SyncedMemory mid2 = mul_mod(mid1, mid1);
        mid1.~SyncedMemory();
        SyncedMemory mid3 = sub_mod(mid2, a);
        mid2.~SyncedMemory();
        SyncedMemory mid4 = sub_mod(mid3, c);
        mid3.~SyncedMemory();
        SyncedMemory d = add_mod(mid4, mid4);
        mid4.~SyncedMemory();

        // E = 3*A
        SyncedMemory mid5 = add_mod(a, a);
        SyncedMemory e = add_mod(mid5, a);
        mid5.~SyncedMemory();
        // F = E^2
        SyncedMemory f = mul_mod(e, e);

        // Z3 = 2*Y1*Z1
        SyncedMemory mid6 = mul_mod(self.y, self.z);
        SyncedMemory z = add_mod(mid6, mid6);
        mid6.~SyncedMemory();
        // X3 = F-2*D
        SyncedMemory mid7 = sub_mod(f, d);
        SyncedMemory x = sub_mod(mid7, d);
        mid7.~SyncedMemory();
        // Y3 = E*(D-X3)-8*C
        SyncedMemory mid8 = sub_mod(d, x);
        SyncedMemory mid9 = add_mod(c, c);
        SyncedMemory mid10 = add_mod(mid9, mid9);
        mid9.~SyncedMemory();
        SyncedMemory mid11 = add_mod(mid10, mid10);
        mid10.~SyncedMemory();
        SyncedMemory mid12 = mul_mod(e, mid8);
        mid8.~SyncedMemory();
        SyncedMemory y = sub_mod(mid12, mid11);

        return ProjectivePointG1(x, y, z);
    }
}
ProjectivePointG1 add_assign_mixed(ProjectivePointG1 self, AffinePointG1 other){
    if (AffinePointG1::is_zero(other)){
        SyncedMemory x(self.x.size(), false);
        SyncedMemory y(self.y.size(), false);
        SyncedMemory z(self.z.size(), false);

        void* x_ = x.mutable_cpu_data();
        void* y_ = y.mutable_cpu_data();
        void* z_ = z.mutable_cpu_data();

        void* self_x_ = self.x.mutable_cpu_data();
        void* self_y_ = self.y.mutable_cpu_data();
        void* self_z_ = self.z.mutable_cpu_data();

        memcpy(x_, self_x_, x.size());
        memcpy(y_, self_y_, y.size());
        memcpy(z_, self_z_, z.size());

        return ProjectivePointG1(x,y,z);
    }
    else if (ProjectivePointG1::is_zero(self)){
        SyncedMemory x(other.x.size(), false);
        SyncedMemory y(other.y.size(), false);

        void* x_ = x.mutable_cpu_data();
        void* y_ = y.mutable_cpu_data();

        void* self1_x_ = other.x.mutable_cpu_data();
        void* self1_y_ = other.y.mutable_cpu_data();

        memcpy(x_, self1_x_, x.size());
        memcpy(y_, self1_y_, y.size());

        //z = self.z.one()  // Assuming z.one() is a method to get a representation of one.
        SyncedMemory z = fq::one();
        return ProjectivePointG1(x,y,z);
    }
    else{
        // Z1Z1 = Z1^2
        SyncedMemory z1z1 = mul_mod(self.z, self.z);

        // U2 = X2*Z1Z1
        SyncedMemory u2 = mul_mod(other.x, z1z1);

        // S2 = Y2*Z1*Z1Z1
        SyncedMemory s2 = mul_mod(mul_mod(other.y, self.z), z1z1);

        if (fq::is_equal(self.x, u2) && fq::is_equal(self.y, s2)){
            // The two points are equal, so we double.
            return double_ProjectivePointG1(self);
        }
        else{
            // H = U2-X1
            SyncedMemory h = sub_mod(u2, self.x);

            // I = 4*(H^2)
            SyncedMemory mid1 = mul_mod(h, h);
            SyncedMemory mid2 = add_mod(mid1, mid1);
            SyncedMemory i = add_mod(mid2, mid2);
            mid1.~SyncedMemory();
            mid2.~SyncedMemory();
            // J = H*I
            SyncedMemory j = mul_mod(h, i);

            // r = 2*(S2-Y1)
            SyncedMemory mid3 = sub_mod(s2, self.y);
            SyncedMemory r = add_mod(mid3, mid3);
            mid3.~SyncedMemory();
            // V = X1*I
            SyncedMemory v = mul_mod(self.x, i);

            // X3 = r^2 - J - 2*V
            SyncedMemory mid4 = sub_mod(mul_mod(r, r), j);
            SyncedMemory v2 = add_mod(v, v);
            SyncedMemory x = sub_mod(mid4, v2);
            mid4.~SyncedMemory();
            // Y3 = r*(V-X3) - 2*Y1*J
            SyncedMemory mid5 = mul_mod(r, sub_mod(v, x));
            SyncedMemory s1j = mul_mod(self.y, j);
            SyncedMemory s1j2 = add_mod(s1j, s1j);
            SyncedMemory y = sub_mod(mid5, s1j2);
            mid5.~SyncedMemory();
            // Z3 = (Z1+H)^2 - Z1Z1 - H^2
            SyncedMemory mid6 = add_mod(self.z, h);
            SyncedMemory mid7 = mul_mod(mid6, mid6);
            SyncedMemory hh = mul_mod(h, h);
            SyncedMemory z = sub_mod(sub_mod(mid7, z1z1), hh);
            mid6.~SyncedMemory();
            mid7.~SyncedMemory();
            
            return ProjectivePointG1(x, y, z);
        }
    }
}