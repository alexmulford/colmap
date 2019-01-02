#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/option_manager.h"
#include <pcl/common/eigen.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <Eigen/Dense>
#include <math.h>
#include <iomanip>
#include <sstream>
#include <boost/thread/thread.hpp>
#include <proj_api.h>
using namespace colmap;

#define IS_BIG_ENDIAN (*(uint16_t *)"\0\xff" < 0x100)

struct Vec3
{
    Vec3(double x = 0.0, double y = 0.0, double z = 0.0);
    Vec3(const Vec3 &o);
    
    double x_,y_,z_;    /**< The x, y and z values of the vector. **/
    
    /*!
      * \brief cross     The cross product between two vectors.
      **/
    Vec3 cross(Vec3 o) const;
    
    /*!
      * \brief dot     The scalar product between two vectors.
      **/
    double dot(Vec3 o) const;
    
    /*!
      * \brief length     The length of the vector.
      **/
    double length() const;
    
    /*!
      * \brief norm     Returns a normalized version of this vector.
      **/
    Vec3 norm() const;
    
    /*!
      * \brief Scales this vector.
      **/
    Vec3 operator*(double d) const;
    
    /*!
      * \brief Addition between two vectors.
      **/
    Vec3 operator+(Vec3 o) const;
    
    /*!
      * \brief Subtraction between two vectors.
      **/
    Vec3 operator-(Vec3 o) const;

    friend std::ostream & operator<<(std::ostream &os, Vec3 v)
    {
        return os << "[" << std::setprecision(8) << v.x_ << ", " << std::setprecision(4) << v.y_ << ", " << v.z_ << "]";
    }
};

class OnMat3
{
  public:
    OnMat3(Vec3 r1, Vec3 r2, Vec3 r3);
    OnMat3(const OnMat3 &o);

    Vec3 r1_;   /**< The first row of the matrix. **/
    Vec3 r2_;   /**< The second row of the matrix. **/
    Vec3 r3_;   /**< The third row of the matrix. **/
    Vec3 c1_;   /**< The first column of the matrix. **/
    Vec3 c2_;   /**< The second column of the matrix. **/
    Vec3 c3_;   /**< The third column of the matrix. **/
    
    /*!
      * \brief The determinant of the matrix.
      **/
    double det() const;
    
    /*!
      * \brief The transpose of the OnMat3 (equal to inverse).
      **/
    OnMat3 transpose() const;
    
    /*!
      * \brief Matrix multiplication between two ON matrices.
      **/ 
    OnMat3 operator*(OnMat3 o) const;
    
    /*!
      * \brief Right side multiplication with a 3d vector.
      **/ 
    Vec3 operator*(Vec3 o);
    
    friend std::ostream & operator<<(std::ostream &os, OnMat3 m)
    {
        return os << "[" << std::endl << m.r1_ << std::endl << m.r2_ << std::endl << m.r3_ << std::endl << "]" << std::endl;
    }
};

class Mat4
{
  public:
    Mat4();
    Mat4(OnMat3 rotation, Vec3 translation, double scaling);
    
    /*!
      * \brief Right side multiplication with a 3d vector.
      **/ 
    Vec3 operator*(Vec3 o);
    
    double r1c1_;   /**< Matrix element 0 0 **/
    double r1c2_;   /**< Matrix element 0 1 **/
    double r1c3_;   /**< Matrix element 0 2 **/
    double r1c4_;   /**< Matrix element 0 3 **/
    double r2c1_;   /**< Matrix element 1 0 **/
    double r2c2_;   /**< Matrix element 1 1 **/
    double r2c3_;   /**< Matrix element 1 2 **/
    double r2c4_;   /**< Matrix element 1 3 **/
    double r3c1_;   /**< Matrix element 2 0 **/
    double r3c2_;   /**< Matrix element 2 1 **/
    double r3c3_;   /**< Matrix element 2 2 **/
    double r3c4_;   /**< Matrix element 2 3 **/
    double r4c1_;   /**< Matrix element 3 0 **/
    double r4c2_;   /**< Matrix element 3 1 **/
    double r4c3_;   /**< Matrix element 3 2 **/
    double r4c4_;   /**< Matrix element 3 3 **/
    
    friend std::ostream & operator<<(std::ostream &os, Mat4 m)
    {
        std::stringstream ss;
        ss.precision(8);
        ss.setf(std::ios::fixed, std::ios::floatfield);
        
        ss << "[ " << m.r1c1_ << ",\t" << m.r1c2_ << ",\t" << m.r1c3_ << ",\t" << m.r1c4_ << " ]" << std::endl << 
              "[ " << m.r2c1_ << ",\t" << m.r2c2_ << ",\t" << m.r2c3_ << ",\t" << m.r2c4_ << " ]" << std::endl << 
              "[ " << m.r3c1_ << ",\t" << m.r3c2_ << ",\t" << m.r3c3_ << ",\t" << m.r3c4_ << " ]" << std::endl << 
              "[ " << m.r4c1_ << ",\t" << m.r4c2_ << ",\t" << m.r4c3_ << ",\t" << m.r4c4_ << " ]";
        
        return os << ss.str();
    }
    
};

class FindTransform
{
public:
    /*!
      * \brief findTransform    Generates an affine transform from the three 'from' vector to the three 'to' vectors.
      *                         The transform is such that transform * fromA = toA,
      *                                                    transform * fromB = toB,
      *                                                    transform * fromC = toC,
      **/ 
    void findTransform(Vec3 fromA, Vec3 fromB, Vec3 fromC, Vec3 toA, Vec3 toB, Vec3 toC);
    
    /*!
      * \brief error     Returns the distance beteween the 'from' and 'to' vectors, after the transform has been applied.
      **/ 
    double error(Vec3 fromA, Vec3 toA);
    
    Mat4 transform_;    /**< The affine transform. **/
};

Vec3::Vec3(double x, double y, double z) :x_(x), y_(y), z_(z)
{}
Vec3::Vec3(const Vec3 &o) : x_(o.x_), y_(o.y_), z_(o.z_)
{}

Vec3 Vec3::cross(Vec3 o) const
{
    Vec3 res;
    res.x_ = y_*o.z_ - z_*o.y_;
    res.y_ = z_*o.x_ - x_*o.z_;
    res.z_ = x_*o.y_ - y_*o.x_;
    return res;
}

double Vec3::dot(Vec3 o) const
{
    return x_*o.x_ + y_*o.y_ + z_*o.z_;
}

double Vec3::length() const
{
    return sqrt(x_*x_ + y_*y_ + z_*z_);
}

Vec3 Vec3::norm() const
{
    Vec3 res;
    double l = length();
    res.x_ = x_ / l;
    res.y_ = y_ / l;
    res.z_ = z_ / l;
    return res;
}

Vec3 Vec3::operator*(double d) const
{
    return Vec3(x_*d, y_*d, z_*d);
}

Vec3 Vec3::operator+(Vec3 o) const
{
    return Vec3(x_ + o.x_, y_ + o.y_,z_ + o.z_);
}

Vec3 Vec3::operator-(Vec3 o) const
{
    return Vec3(x_ - o.x_, y_ - o.y_,z_ - o.z_);
}

OnMat3::OnMat3(Vec3 r1, Vec3 r2, Vec3 r3) : r1_(r1), r2_(r2), r3_(r3)
{
    c1_.x_ = r1_.x_; c2_.x_ = r1_.y_; c3_.x_ = r1_.z_;
    c1_.y_ = r2_.x_; c2_.y_ = r2_.y_; c3_.y_ = r2_.z_;
    c1_.z_ = r3_.x_; c2_.z_ = r3_.y_; c3_.z_ = r3_.z_;
}
OnMat3::OnMat3(const OnMat3 &o) : r1_(o.r1_), r2_(o.r2_), r3_(o.r3_)
{
    c1_.x_ = r1_.x_; c2_.x_ = r1_.y_; c3_.x_ = r1_.z_;
    c1_.y_ = r2_.x_; c2_.y_ = r2_.y_; c3_.y_ = r2_.z_;
    c1_.z_ = r3_.x_; c2_.z_ = r3_.y_; c3_.z_ = r3_.z_;
}

double OnMat3::det() const
{
    return r1_.x_*r2_.y_*r3_.z_ + r1_.y_*r2_.z_*r3_.x_ + r1_.z_*r2_.x_*r3_.y_ - r1_.z_*r2_.y_*r3_.x_ - r1_.y_*r2_.x_*r3_.z_ - r1_.x_*r2_.z_*r3_.y_;
}

OnMat3 OnMat3::transpose() const
{
    return OnMat3(Vec3(r1_.x_, r2_.x_, r3_.x_), Vec3(r1_.y_, r2_.y_, r3_.y_), Vec3(r1_.z_, r2_.z_, r3_.z_));
}

OnMat3 OnMat3::operator*(OnMat3 o) const
{
    return OnMat3(  Vec3(r1_.dot(o.c1_), r1_.dot(o.c2_), r1_.dot(o.c3_)),
                    Vec3(r2_.dot(o.c1_), r2_.dot(o.c2_), r2_.dot(o.c3_)),
                    Vec3(r3_.dot(o.c1_), r3_.dot(o.c2_), r3_.dot(o.c3_)));
}

Vec3 OnMat3::operator*(Vec3 o)
{
    return Vec3(r1_.dot(o), r2_.dot(o), r3_.dot(o));
}

Mat4::Mat4()
{
    r1c1_ = 1.0;  r1c2_ = 0.0;  r1c3_ = 0.0;  r1c4_ = 0.0;
    r2c1_ = 0.0;  r2c2_ = 1.0;  r2c3_ = 0.0;  r2c4_ = 0.0;
    r3c1_ = 0.0;  r3c2_ = 0.0;  r3c3_ = 1.0;  r3c4_ = 0.0;
    r4c1_ = 0.0;  r4c2_ = 0.0;  r4c3_ = 0.0;  r4c4_ = 1.0;
}

Mat4::Mat4(OnMat3 rotation, Vec3 translation, double scaling)
{
    r1c1_ = scaling * rotation.r1_.x_;  r1c2_ = scaling * rotation.r1_.y_;  r1c3_ = scaling * rotation.r1_.z_;  r1c4_ = translation.x_;
    r2c1_ = scaling * rotation.r2_.x_;  r2c2_ = scaling * rotation.r2_.y_;  r2c3_ = scaling * rotation.r2_.z_;  r2c4_ = translation.y_;
    r3c1_ = scaling * rotation.r3_.x_;  r3c2_ = scaling * rotation.r3_.y_;  r3c3_ = scaling * rotation.r3_.z_;  r3c4_ = translation.z_;
    r4c1_ = 0.0;            r4c2_ = 0.0;            r4c3_ = 0.0;            r4c4_ = 1.0;
}

Vec3 Mat4::operator*(Vec3 o)
{
    return Vec3(
                r1c1_ * o.x_ + r1c2_* o.y_ + r1c3_* o.z_ + r1c4_,
                r2c1_ * o.x_ + r2c2_* o.y_ + r2c3_* o.z_ + r2c4_,
                r3c1_ * o.x_ + r3c2_* o.y_ + r3c3_* o.z_ + r3c4_
                );
}

void FindTransform::findTransform(Vec3 fromA, Vec3 fromB, Vec3 fromC, Vec3 toA, Vec3 toB, Vec3 toC)
{
    Vec3 a1 = toA;
    Vec3 b1 = toB;
    Vec3 c1 = toC;
    Vec3 a2 = fromA;
    Vec3 b2 = fromB;
    Vec3 c2 = fromC;
        
    Vec3 y1 = (a1 - c1).cross(b1 - c1).norm();
    Vec3 z1 = (a1 - c1).norm();
    Vec3 x1 = y1.cross(z1);
    
    Vec3 y2 = (a2 - c2).cross(b2 - c2).norm();
    Vec3 z2 = (a2 - c2).norm();
    Vec3 x2 = y2.cross(z2);
    OnMat3 mat1 = OnMat3(x1, y1, z1).transpose();
    OnMat3 mat2 = OnMat3(x2, y2, z2).transpose();
    
    OnMat3 rotation = mat1 * mat2.transpose();
    Vec3 translation = c1 - c2;
    
    double scale = (a1 - c1).length() / (a2 - c2).length();
    
    translation = rotation * c2 * (-scale) + c1;
    Mat4 transformation(rotation, translation, scale);
    transform_ = transformation;
}

double FindTransform::error(Vec3 fromA, Vec3 toA)
{
    return (transform_*fromA - toA).length();
}



struct GeorefBestTriplet
{
    size_t t_;          /**< First ordinate of the best triplet found. **/
    size_t s_;          /**< Second ordinate of the best triplet found. **/
    size_t p_;          /**< Third ordinate of the best triplet found. **/
    double err_;        /**< Error of this triplet. **/
};

class GeoRef
{
  public:
  std::vector<Vec3> local_ts;
  std::vector<Vec3> gps_t;
  GeoRef(std::vector<Vec3> local_ts,std::vector<Vec3> gps_t)
  {
    this->local_ts = local_ts;
    this->gps_t = gps_t;
  }

  void findBestCameraTriplet(size_t &cam0, size_t &cam1, size_t &cam2, size_t offset, size_t stride, double &minTotError);
  void chooseBestCameraTriplet(size_t &cam0, size_t &cam1, size_t &cam2);
};

void GeoRef::findBestCameraTriplet(size_t &cam0, size_t &cam1, size_t &cam2, size_t offset, size_t stride,double &minTotError)
{
    minTotError = std::numeric_limits<double>::infinity();
    
    for(size_t t = offset; t < local_ts.size(); t+=stride)
    {
        for(size_t s = t; s < local_ts.size(); ++s)
        {
            for(size_t p = s; p < local_ts.size(); ++p)
            {
                FindTransform trans;
                //easting_,northing_,altitude_
                trans.findTransform(local_ts[t], local_ts[s], local_ts[p],gps_t[t], gps_t[s], gps_t[p]);
                
                // The total error for the current camera triplet.
                double totError = 0.0;
                
                for(size_t r = 0; r < gps_t.size(); ++r)
                {
                    totError += trans.error(local_ts[r], gps_t[r]);
                }
                
                if(minTotError > totError)
                {
                    minTotError = totError;
                    cam0 = t;
                    cam1 = s;
                    cam2 = p;
                }
            }
        }
    }
}

void GeoRef::chooseBestCameraTriplet(size_t &cam0, size_t &cam1, size_t &cam2)
{
    size_t numThreads = boost::thread::hardware_concurrency();
    boost::thread_group threads;

    // To be changed
    std::vector<GeorefBestTriplet*> triplets;
    for(size_t t = 0; t < numThreads; ++t)
    {
        GeorefBestTriplet* triplet = new GeorefBestTriplet();
        triplets.push_back(triplet);
        threads.create_thread(boost::bind(&GeoRef::findBestCameraTriplet, this, boost::ref(triplet->t_), boost::ref(triplet->s_), boost::ref(triplet->p_), t, numThreads, boost::ref(triplet->err_)));
    }

    threads.join_all();

    double minTotError = std::numeric_limits<double>::infinity();
    for(size_t t = 0; t<numThreads; t++)
    {
        GeorefBestTriplet* triplet = triplets[t];
        if(minTotError > triplet->err_)
        {
            minTotError = triplet->err_;
            cam0 = triplet->t_;
            cam1 = triplet->s_;
            cam2 = triplet->p_;
        }
        delete triplet;
    }
}

template <typename Scalar>
void transformPointCloud(const char *inputFile, const Eigen::Transform<Scalar, 3, Eigen::Affine> &transform, const char *outputFile){
    try{
        
        pcl::PointCloud<pcl::PointXYZRGBNormal> pointCloud;
        if(pcl::io::loadPLYFile<pcl::PointXYZRGBNormal> (inputFile, pointCloud) == -1) {
            std::cout << "Error when reading point cloud:\n" + std::string(inputFile) + "\n" <<std::endl;
            return;
        }
        else
        {
            std::cout << "Successfully loaded " << pointCloud.size() << " points with corresponding normals from file.\n"<<std::endl;
        }
        std::cout << "Writing transformed point cloud to " << outputFile << "...\n"<<std::endl;
        // We don't use PCL's built-in functions
        // because PCL does not support double coordinates
        // precision
        std::ofstream f (outputFile);
        f << "ply" << std::endl;
        if (IS_BIG_ENDIAN){
          f << "format binary_big_endian 1.0" << std::endl;
        }else{
          f << "format binary_little_endian 1.0" << std::endl;
        }
        const char *type = "double";
        if (sizeof(Scalar) == sizeof(float)){
            type = "float";
        }
        f   << "element vertex " << pointCloud.size() << std::endl
            << "property " << type << " x" << std::endl
            << "property " << type << " y" << std::endl
            << "property " << type << " z" << std::endl
            << "property " << type << " nx" << std::endl
            << "property " << type << " ny" << std::endl
            << "property " << type << " nz" << std::endl
            << "property uchar red" << std::endl
            << "property uchar green" << std::endl
            << "property uchar blue" << std::endl
            << "end_header" << std::endl;
        struct PlyPoint{
            Scalar x;
            Scalar y;
            Scalar z;
            Scalar nx;
            Scalar ny;
            Scalar nz;
            uint8_t r;
            uint8_t g;
            uint8_t b;
        } p;
        size_t psize = sizeof(Scalar) * 6 + sizeof(uint8_t) * 3;
        //For normal transformation
        Eigen::MatrixXd b(4,4);
        b << transform (0, 0),transform (0, 1),transform (0, 2), transform (0, 3),
        transform (1, 0),transform (1, 1),transform (1, 2), transform (1, 3),
        transform (2, 0),transform (2, 1),transform (2, 2), transform (2, 3),
        transform (3, 0),transform (3, 1),transform (3, 2), transform (3, 3);
        b = b.inverse();
        b = b.transpose();
        // Transform
        for (unsigned int i = 0; i < pointCloud.size(); i++){
            Scalar x = static_cast<Scalar>(pointCloud[i].x);
            Scalar y = static_cast<Scalar>(pointCloud[i].y);
            Scalar z = static_cast<Scalar>(pointCloud[i].z);
            Scalar tnx = static_cast<Scalar>(pointCloud[i].normal_x);
            Scalar tny = static_cast<Scalar>(pointCloud[i].normal_y);
            Scalar tnz = static_cast<Scalar>(pointCloud[i].normal_z);
            p.r = pointCloud[i].r;
            p.g = pointCloud[i].g;
            p.b = pointCloud[i].b;
            p.x = static_cast<Scalar> (transform (0, 0) * x + transform (0, 1) * y + transform (0, 2) * z + transform (0, 3));
            p.y = static_cast<Scalar> (transform (1, 0) * x + transform (1, 1) * y + transform (1, 2) * z + transform (1, 3));
            p.z = static_cast<Scalar> (transform (2, 0) * x + transform (2, 1) * y + transform (2, 2) * z + transform (2, 3));
            p.nx = static_cast<Scalar> (b (0, 0) * tnx + b (0, 1) * tny + b (0, 2) * tnz + b (0, 3));
            p.ny = static_cast<Scalar> (b (1, 0) * tnx + b (1, 1) * tny + b (1, 2) * tnz + b (1, 3));
            p.nz = static_cast<Scalar> (b (2, 0) * tnx + b (2, 1) * tny + b (2, 2) * tnz + b (2, 3));
            f.write(reinterpret_cast<char*>(&p), psize);
            // TODO: normals can be computed using the inverse transpose
            // https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
        }
        f.close();
        std::cout << "Point cloud file saved.\n" << std::endl;
    }
    catch (const std::exception & e)
    {
        std::cout << "Error while loading point cloud: " + std::string(e.what())<< std::endl;
        return;
    }
}

void convertGPS2UTM(const double &lon, const double &lat, const double &alt, double &x, double &y, double &z, int &utmZone, char &hemisphere)
{
  // Create WGS84 ecef coordinate system
  projPJ pjLatLon = pj_init_plus("+proj=latlong +datum=WGS84");
  if (!pjLatLon) {
    std::cout << ("Couldn't create WGS84 coordinate system with PROJ.4.") << std::endl;
    return;
  }

  // Calculate UTM zone if it's set to magic 99
  // NOTE: Special UTM cases in Norway/Svalbard not supported here
  if (utmZone == 99) {
    utmZone = ((static_cast<int>(floor((lon + 180.0)/6.0)) % 60) + 1);
    if (lat < 0)
      hemisphere = 'S';
    else
      hemisphere = 'N';
  }

  std::ostringstream ostr;
  ostr << utmZone;
  if (hemisphere == 'S')
    ostr << " +south";

  // Create UTM coordinate system
  projPJ pjUtm = pj_init_plus(("+proj=utm +datum=WGS84 +zone=" + ostr.str()).c_str());
  if (!pjUtm) {
    std::cout << ("Couldn't create UTM coordinate system with PROJ.4.") << std::endl;
    return;
  }

  // Convert to radians
  x = lon * 0.017453292;
  y = lat * 0.017453292;
  z = alt;

  // Transform
  int res = pj_transform(pjLatLon, pjUtm, 1, 1, &x, &y, &z);
  if (res != 0) {
    std::cout << "Failed to transform coordinates" << std::endl;
    return;
  }
}

// Simple example that reads and writes a reconstruction.
int main(int argc, char** argv) {
  InitializeGlog(argv);
  /********************************Load all the information**************************************/
  std::string input_path;
  std::string input_ply_path;
  std::string input_GNSS_path;
  std::string output_ply_path;
  bool use_utm = false;
  int zone = 31;
  char sphere = 'N';
  bool addOffset = false;
  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("input_ply_path", &input_ply_path);
  options.AddRequiredOption("input_GNSS_path", &input_GNSS_path);
  options.AddRequiredOption("output_ply_path", &output_ply_path);
  options.AddRequiredOption("use_utm",&use_utm);
  options.AddDefaultOption("add_offset",&addOffset);
  options.AddDefaultOption("utm_zone", &zone);
  options.AddDefaultOption("utm_hemisphere", &sphere);
  options.Parse(argc, argv);
  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  //All vectors
  std::vector<std::string> image_names;
  std::vector<double> qws, qxs, qys, qzs;
  std::vector<Vec3> local_ts;
  //Get all images from model
  std::vector<image_t> reg_ids = reconstruction.RegImageIds();
  for (uint i = 0; i < reg_ids.size(); i++)
  {
    image_names.push_back(reconstruction.Image(reg_ids[i]).Name());
    qws.push_back(reconstruction.Image(reg_ids[i]).Qvec()[0]);
    qxs.push_back(reconstruction.Image(reg_ids[i]).Qvec()[1]);
    qys.push_back(reconstruction.Image(reg_ids[i]).Qvec()[2]);
    qzs.push_back(reconstruction.Image(reg_ids[i]).Qvec()[3]);
    double tx = reconstruction.Image(reg_ids[i]).Tvec()[0];
    double ty = reconstruction.Image(reg_ids[i]).Tvec()[1];
    double tz = reconstruction.Image(reg_ids[i]).Tvec()[2];
    Vec3 temp(tx,ty,tz);
    local_ts.push_back(temp);
  }
  //Load ECEF files
  std::vector<std::string> ecef_names;
  std::vector<Vec3> gps_t(reg_ids.size(),Vec3(0.0,0.0,0.0));
  std::ifstream myfile (input_GNSS_path);
  std::string stemp;
  uint counter = 0;
  if (myfile.is_open())
  {
    //Read line by line
    while (getline(myfile,stemp))
    {
      std::vector<std::string> strs;
      boost::split(strs,stemp,boost::is_any_of(" "));
      //Search Weather existed in reconstruction
      for(uint j = 0; j < image_names.size();j++)
      {
        if (strs[0] == image_names[j])
        {
          ecef_names.push_back(strs[0]);
          counter++;
          double gps_x = std::stod(strs[1]);
          double gps_y = std::stod(strs[2]);
          double gps_z = std::stod(strs[3]);
          Vec3 gps_temp(gps_x,gps_y,gps_z);
          gps_t[j]=gps_temp;
        }
      }
    }
    myfile.close();
  }
  else
  {
    std::cout << "GNSS File not found, Exiting......" <<std::endl;
    return 0;
  }
  if (counter == reg_ids.size())
  {
    std::cout << "ALL Images have corresponding GNSS! " << counter << " images in total!"<< std::endl;
  }
  else
  {
    std::cout << "Not All Images have corresponding GNSS!, Filtering out." << std::endl;
    //For each reconstruction image
    for (uint k=0;k<reg_ids.size();k++)
    {
      if ((gps_t[k].x_==0.0) && (gps_t[k].y_==0.0) && (gps_t[k].z_==0.0))
      {
        gps_t.erase(gps_t.begin()+k);
        local_ts.erase(local_ts.begin()+k);
      }
    }
    std::cout << "The vector size is :" << local_ts.size() << " vs " << gps_t.size() << std::endl;
  }

  /************************************PreProcessing-Centering****************************************/
  if (use_utm)
  {

    std::cout << "Using UTM, converting all GPS to UTM, using: zone: " << zone << " and hemisphere: " << sphere << std::endl;


    for (std::vector<Vec3>::iterator iter = gps_t.begin(); iter != gps_t.end(); ++iter)
    {
      const double lon = iter->x_;
      const double lat = iter->y_;
      const double alt = iter->z_;
      double x, y ,z;
      convertGPS2UTM(lon,lat,alt,x,y,z,zone,sphere);
      iter->x_ = x;
      iter->y_ = y;
      iter->z_ = z;
    }
  }
  double dx = 0.0, dy = 0.0,dz=0.0;
  double num = static_cast<double>(counter);
  for (std::vector<Vec3>::iterator iter = gps_t.begin(); iter != gps_t.end(); ++iter) {
    dx += iter->x_/num;
    dy += iter->y_/num;
    dz += iter->z_/num;
  }
  dx = floor(dx);
  dy = floor(dy);
  dz = floor(dz);

  std::ofstream geo_info_file;
  geo_info_file.open(output_ply_path+"_Geo_info.txt");
  std::string info_line;

  if (use_utm)
  {
    info_line = ("UTM: Zone: " + std::to_string(zone) + " Hemisphere: " + sphere + " Northing: " + std::to_string(dx) + " Easting: " + std::to_string(dy) + " Aviation: " + std::to_string(dz));
  }
  else
  {
    info_line = ("ECEF: Center_x: " + std::to_string(dx) + " Center_y: " + std::to_string(dy) + " Center_z: " + std::to_string(dz)); 
  }
  geo_info_file<<info_line;
  geo_info_file.close();


  std::cout << "Center point (Offset) at: " << dx << " " << dy<< " " << dz << std::endl;
  for (std::vector<Vec3>::iterator iter = gps_t.begin(); iter != gps_t.end(); ++iter) {
    iter->x_ -= dx;
    iter->y_ -= dy;
    iter->z_ -= dz;
  }

  /*************************************Choosing the best cameras************************************/
  size_t cam0,cam1,cam2;
  GeoRef georef(local_ts,gps_t);
  georef.chooseBestCameraTriplet(cam0,cam1,cam2);
  std::cout << "The best triplet is chosen as: " << cam0 << " " << cam1 << " " << cam2 << std::endl;
  FindTransform transFinal;
  transFinal.findTransform(local_ts[cam0], local_ts[cam1], local_ts[cam2],gps_t[cam0], gps_t[cam1], gps_t[cam2]);
  Mat4 transMat = transFinal.transform_;
  std::cout<< transFinal.transform_ << std::endl;
  Eigen::Transform<double, 3, Eigen::Affine> transform;
  transform(0, 0) = static_cast<double>(transMat.r1c1_);
  transform(1, 0) = static_cast<double>(transMat.r2c1_);
  transform(2, 0) = static_cast<double>(transMat.r3c1_);
  transform(3, 0) = static_cast<double>(transMat.r4c1_);
  transform(0, 1) = static_cast<double>(transMat.r1c2_);
  transform(1, 1) = static_cast<double>(transMat.r2c2_);
  transform(2, 1) = static_cast<double>(transMat.r3c2_);
  transform(3, 1) = static_cast<double>(transMat.r4c2_);
  transform(0, 2) = static_cast<double>(transMat.r1c3_);
  transform(1, 2) = static_cast<double>(transMat.r2c3_);
  transform(2, 2) = static_cast<double>(transMat.r3c3_);
  transform(3, 2) = static_cast<double>(transMat.r4c3_);
  transform(0, 3) = static_cast<double>(transMat.r1c4_);
  transform(1, 3) = static_cast<double>(transMat.r2c4_);
  transform(2, 3) = static_cast<double>(transMat.r3c4_);
  transform(3, 3) = static_cast<double>(transMat.r4c4_);


  if (addOffset){
    transform(0, 3) = transform(0, 3) + dx;
    transform(1, 3) = transform(1, 3) + dy;
    transform(2, 3) = transform(2, 3) + dz;
  }

  transformPointCloud(input_ply_path.c_str(),transform,output_ply_path.c_str());
}
