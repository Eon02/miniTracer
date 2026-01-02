//
// Created by haley on 9/2/25.
//
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <eigen-3.4.0/Eigen/Eigen>
#include <cmath>

using namespace std;
   struct point {
       float x = 0;
       float y = 0;
       float z = 0;

       point(float x, float y, float z) : x(x), y(y), z(z) {}
       point() {
           x = 0;
           y = 0;
           z = 0;
       };


       point operator-(const point &p) {
           return point(x - p.x, y - p.y, z - p.z);
       }
       point operator+(const point &p) const {
           return point(x + p.x, y + p.y, z + p.z);
       }
       point operator*(const float &f) {
           return point(x * f, y * f, z * f);
       }
       point operator/(const float &f) {
           return point(x / f, y / f, z / f);
       }
      bool operator<(const point &p) {
           if ( x < p.x || y < p.y || z < p.z) {
               return true;
           }
           return false;
       }
       bool operator>(const point &p) {
           if ( x > p.x || y > p.y || z > p.z) {
               return true;
           }
           return false;
       }
       bool operator==(const point &p) {
           if ( x == p.x && y == p.y && z == p.z) {
               return true;
           }
           return false;
       }
       bool operator!=(const point &p) {
           if ( x != p.x || y != p.y || z != p.z) {
               return true;
           }
           return false;
       }


   };
    struct pixel {
        int r = 100;
        int g = 100;
        int b = 100;
        pixel(int r, int g, int b) : r(r), g(g), b(b) {}
        pixel() {
            r = 100;
            g = 100;
            b = 100;
        }

    };

class ray {
    public:
        point e = point(0,0,0);
        Eigen::Vector3f dir;
        point endPoint = point(0,0,0);


    ray(float ex, float ey, float ez, float xdir, float ydir, float zdir) {
            e = point(ex, ey, ez);
            dir = Eigen::Vector3f(xdir, ydir, zdir);
        }
};
class eyePoint {
    point pos = point(0, 0, 0);
    ray u = ray(0,0,0,1,0,0);
    ray v = ray(0,0,0,0,0,1);
    ray w =  ray(0,0,0,0,-1,0);
    eyePoint();
    eyePoint(float ex, float ey, float ez, float xdir, float ydir, float zdir) {
        pos = point(ex, ey, ez);
        u = ray(ex,ey,ez,xdir,0,0);
        v = ray(ex,ey,ez,0,ydir,0);
        w = ray(ex,ey,ez,0,0,zdir);

    }
};


class image {
    public:
        int width=512;
        int height=512;
        vector<unsigned char> pixels;
    image() {
        for (int i=0; i< width*height*3; i++) {
                pixels.push_back(100);
        }
    }
    image(int w, int h)  {
            width = w;
            height = h;
            for (int i=0; i< width*height*3; i++) {
                pixels.push_back(100);
            }
        }
};
class orthoCamera {
public:
    image img;
    vector<ray> eyeRays;
    point pos = point(0,0,0);
    Eigen::Vector3d lookDir = Eigen::Vector3d(0,1,0);
    
    // Camera coordinate system vectors
    Eigen::Vector3d right;
    Eigen::Vector3d up;
    Eigen::Vector3d forward;
    
    orthoCamera(): img(512, 512) {
        setupCamera(512, 512, 0, 0, 0, Eigen::Vector3d(0,1,0));
    }
    
    orthoCamera(int w, int h, float posx, float posy, float posz, Eigen::Vector3d dir): img(w, h) {
        setupCamera(w, h, posx, posy, posz, dir);
    }
    
private:
    void setupCamera(int w, int h, float posx, float posy, float posz, Eigen::Vector3d dir) {
        img = image(w, h);
        pos = point(posx, posy, posz);
        lookDir = dir.normalized();

        setupCameraVectors();
        generateOrthoRays(w, h);
    }
    
    void setupCameraVectors() {

        forward = lookDir.normalized();
        Eigen::Vector3d worldUp;
        // if you don't change the world up direction when there is vertical view, the sphere will fill the screen
        // when i looked it up this wsa called gimbal lock ?
        if (abs(forward.z()) > 0.99) {

            worldUp = Eigen::Vector3d(0, 1, 0);
        } else {
            // Otherwise use Z-axis as world up
            worldUp = Eigen::Vector3d(0, 0, 1);
        }



        // right vector is forward cross worldUp -- right hand rule :)
        right = forward.cross(worldUp).normalized();
        
        // cameras up vector is right vector cross worldup
        up = right.cross(forward).normalized();
    }
    
    void generateOrthoRays(int w, int h) {
        eyeRays.clear();
        
        // how many rays will make up the image - may play around w this for lighting?
        float orthoWidth = w;
        float orthoHeight = h;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {

                float u = (col + 0.5f) / w - 0.5f;
                float v = (row + 0.5f) / h - 0.5f;

                u *= orthoWidth;
                v *= orthoHeight;

                //convert uv coordinates to xyz for each ray: e + u*u + v*v
                Eigen::Vector3d rayOrigin = Eigen::Vector3d(pos.x, pos.y, pos.z) + 
                                          u * right + v * up;

                eyeRays.push_back(ray(rayOrigin.x(), rayOrigin.y(), rayOrigin.z(),
                                    forward.x(), forward.y(), forward.z()));
            }
        }
    }
    
public:
    //for changing camera viewpoint
    void updateCamera(float posx, float posy, float posz, Eigen::Vector3d dir) {
        setupCamera(img.width, img.height, posx, posy, posz, dir);
    }

//     void cameraDebug() {
//         std::cout << "Camera Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
//         std::cout << "Look Direction: (" << lookDir.x() << ", " << lookDir.y() << ", " << lookDir.z() << ")\n";
//         std::cout << "Right Vector: (" << right.x() << ", " << right.y() << ", " << right.z() << ")\n";
//         std::cout << "Up Vector: (" << up.x() << ", " << up.y() << ", " << up.z() << ")\n";
//         std::cout << "Forward Vector: (" << forward.x() << ", " << forward.y() << ", " << forward.z() << ")\n";
//     }
 };

class perspectiveCamera {
public:
    image img;
    vector<ray> eyeRays;
    point pos = point(0,0,0);
    Eigen::Vector3d lookDir = Eigen::Vector3d(0,0,1);
    
    // Camera coordinate system vectors
    Eigen::Vector3d right;
    Eigen::Vector3d up;
    Eigen::Vector3d forward;
    float movementSpeed = 80.0f;
    float rotationSpeed = 60.0;

    float yaw = 0.0f;
    float pitch = 0.0f;

    // Perspective-specific parameters
    float fov = 60.0f;  // Field of view in degrees
    float aspectRatio = 1.0f;  // width/height
    
    perspectiveCamera(): img(512, 512) {
        setupCamera(512, 512, 0, 0, 0, Eigen::Vector3d(0,0,1), 60.0f);
    }
    
    perspectiveCamera(int w, int h, float posx, float posy, float posz, 
                     Eigen::Vector3d dir, float fieldOfView = 60.0f): img(w, h) {
        setupCamera(w, h, posx, posy, posz, dir, fieldOfView);
    }
    
private:
    void setupCamera(int w, int h, float posx, float posy, float posz, 
                    Eigen::Vector3d dir, float fieldOfView) {
        img = image(w, h);
        pos = point(posx, posy, posz);
        lookDir = dir.normalized();
        fov = fieldOfView;
        aspectRatio = (float)w / (float)h;

        setupCameraVectors();
        generatePerspectiveRays(w, h);
    }
    
    void setupCameraVectors() {
        forward = lookDir.normalized();
        Eigen::Vector3d worldUp;
        
        // Handle gimbal lock like ortho camera
        if (abs(forward.z()) > 0.99) {

            worldUp = Eigen::Vector3d(0, 1, 0);
        } else {
            worldUp = Eigen::Vector3d(0, 1, 0);
        }

        right = forward.cross(worldUp).normalized();
        up = right.cross(forward).normalized();
    }
    
    void generatePerspectiveRays(int w, int h) {
        eyeRays.clear();
        
        // Calculate the dimensions of the image plane
        float fovRadians = fov * M_PI / 180.0f;
        float imageHeight = 2.0f * tan(fovRadians / 2.0f);
        float imageWidth = imageHeight * aspectRatio;
        
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                // Convert pixel coordinates to normalized device coordinates [-0.5, 0.5]
                float u = (col + 0.5f) / w - 0.5f;
                float v = (row + 0.5f) / h - 0.5f;
                
                // Scale by image plane dimensions
                u *= imageWidth;
                v *= imageHeight;
                
                // Calculate ray direction in world space
                // Ray goes from camera position through the point on the image plane
                Eigen::Vector3d rayDirection = forward + u * right + v * up;
                rayDirection = rayDirection.normalized();
                
                // Camera position is origin for all rays
                eyeRays.push_back(ray(pos.x, pos.y, pos.z,
                                    rayDirection.x(), rayDirection.y(), rayDirection.z()));
            }
        }
    }
    
public:
    void updateCamera(float posx, float posy, float posz, Eigen::Vector3d dir, float fieldOfView = -1) {
        if (fieldOfView > 0) {
            fov = fieldOfView;
        }
        setupCamera(img.width, img.height, posx, posy, posz, dir, fov);
    }
    
    // Debug method
    //     void cameraDebug() {
    //         std::cout << "Perspective Camera Debug:\n";
    //         std::cout << "Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")\n";
    //         std::cout << "Look Direction: (" << lookDir.x() << ", " << lookDir.y() << ", " << lookDir.z() << ")\n";
    //         std::cout << "FOV: " << fov << " degrees\n";
    //         std::cout << "Aspect Ratio: " << aspectRatio << "\n";
    //         std::cout << "Right Vector: (" << right.x() << ", " << right.y() << ", " << right.z() << ")\n";
    //         std::cout << "Up Vector: (" << up.x() << ", " << up.y() << ", " << up.z() << ")\n";
    //         std::cout << "Forward Vector: (" << forward.x() << ", " << forward.y() << ", " << forward.z() << ")\n";
    //     }
    // };
};

class sphere {
public:
    float radius;
    point pos = point(0,0,0);
    pixel color = pixel(255,255,255);

    // Material properties for specular highlights
    float shininess = 32.0f;      // Higher = smaller, sharper highlights
    float specularStrength = 0.5f; // How reflective the surface is (0-1)
    pixel specularColor = pixel(255, 255, 255); // Color of specular highlights

    sphere(float r, float x, float y, float z, int red, int green, int blue) {
        radius = r;
        pos = point(x, y, z);
        color = pixel(red, green, blue);
    }

    // Constructor with material properties
    sphere(float r, float x, float y, float z, int red, int green, int blue,
           float shine, float specStrength) {
        radius = r;
        pos = point(x, y, z);
        color = pixel(red, green, blue);
        shininess = shine;
        specularStrength = specStrength;
    }

    point intersect(ray r) const {
        Eigen::Vector3f V = Eigen::Vector3f(r.e.x - pos.x, r.e.y - pos.y, r.e.z - pos.z);
        Eigen::Vector3f D = r.dir.normalized();
        float t1;

        float discriminant = pow(V.dot(D),2) - (V.dot(V) - radius*radius);

        if (discriminant < 0 ) {
            return point(0,0,0);
        }

        if (discriminant == 0 ) {
            float t = -(V.dot(D));
            point p = point((r.e.x + t*r.dir[0]),(r.e.y + t*r.dir[1]),(r.e.z + t*r.dir[2]));
            r.endPoint = p;
            return p;
        }

        t1 = (-(V.dot(D)) - sqrt(discriminant));
        if (t1< 0) {
            t1 = (-(V.dot(D)) + sqrt(discriminant));
        }
        point p = point((r.e.x + t1*r.dir[0]),(r.e.y + t1*r.dir[1]),(r.e.z + t1*r.dir[2]));
        r.endPoint = p;
        return p;
    }

    // Calculate surface normal at a point on the sphere
    Eigen::Vector3f getNormal(const point& hitPoint) {
        Eigen::Vector3f normal(hitPoint.x - pos.x,
                              hitPoint.y - pos.y,
                              hitPoint.z - pos.z);
        return normal.normalized();
    }
};
class plane {
public:
    point pos;  // A point on the plane
    Eigen::Vector3f normal;  // Plane normal (normalized)
    pixel color = pixel(100, 100, 100);

    // Material properties (like spheres)
    float shininess = 16.0f;
    float specularStrength = 0.3f;
    pixel specularColor = pixel(255, 255, 255);
    float reflectivity;

    // Constructor for a plane defined by a point and normal
    plane(float x, float y, float z, float nx, float ny, float nz,
          int red, int green, int blue) {
        pos = point(x, y, z);
        normal = Eigen::Vector3f(nx, ny, nz).normalized();
        color = pixel(red, green, blue);
        reflectivity = 0.0f;  // No reflection by default
    }

    // Constructor with material properties
    plane(float x, float y, float z, float nx, float ny, float nz,
          int red, int green, int blue, float shine, float specStrength, float ref = 0.0f) {
        pos = point(x, y, z);
        normal = Eigen::Vector3f(nx, ny, nz).normalized();
        color = pixel(red, green, blue);
        shininess = shine;
        specularStrength = specStrength;
        reflectivity  = ref;
    }
    float getReflectivity() const { return reflectivity; }
    void setReflectivity(float refl) { reflectivity = refl; }


    // Ray-plane intersection
    point intersect(const ray& r) const {
        Eigen::Vector3f rayDir = r.dir.normalized();

        // Check if ray is parallel to plane
        float denominator = normal.dot(rayDir);
        if (abs(denominator) < 1e-6) {
            return point(0, 0, 0);  // No intersection (parallel)
        }

        // Calculate intersection distance
        Eigen::Vector3f rayToPlane(pos.x - r.e.x, pos.y - r.e.y, pos.z - r.e.z);
        float t = rayToPlane.dot(normal) / denominator;

        // Check if intersection is behind the ray origin
        if (t < 0) {
            return point(0, 0, 0);  // No intersection (behind ray)
        }

        // Calculate intersection point
        point intersection(r.e.x + t * rayDir.x(),
                          r.e.y + t * rayDir.y(),
                          r.e.z + t * rayDir.z());

        return intersection;
    }

    // Get normal at any point (constant for planes)
    Eigen::Vector3f getNormal(const point& hitPoint) const {
        return normal;
    }

    // Checkerboard! adds a little texture lol
    pixel getColorAt(const point& hitPoint) const {

        int checkSize = 40;  // Size of each checker square
        int checkX = (int)(hitPoint.x / checkSize);
        int checkZ = (int)(hitPoint.z / checkSize);

        if ((checkX + checkZ) % 2 == 0) {
            return color;  // Base color
        } else {
            // Get darker color
            return pixel(color.r * 0.7f, color.g * 0.7f, color.b * 0.7f);
        }
    }
};

class light {
public:
    point pos = point(0,0,0);
    pixel color = pixel(255,255,255);
    float intensity = 1.0f;

    light(float x, float y, float z, int red, int green, int blue, float intens = 1.0f) {
        pos = point(x, y, z);
        color = pixel(red, green, blue);
        intensity = intens;
    }

    // Calculates light direction from a point to the light source
    Eigen::Vector3f getLightDirection(const point& hitPoint) const {
        Eigen::Vector3f lightDir(pos.x - hitPoint.x,
                                pos.y - hitPoint.y,
                                pos.z - hitPoint.z);
        return lightDir.normalized();
    }

    // Calculates distance from point to light
    float getDistance(const point& hitPoint) const {
        return sqrt(pow(pos.x - hitPoint.x, 2) +
                   pow(pos.y - hitPoint.y, 2) +
                   pow(pos.z - hitPoint.z, 2));
    }

    // Calculates light intensity at a given distance (with attenuation)
    float getIntensityAtDistance(float distance) const {
        return intensity / (1.0f + 0.001f * distance * distance);
    }
};


class scene {
public:
    const point origin = point(0,0,0);
    vector<sphere> objects;
    vector<plane> planes;
    vector<light> lights;

    orthoCamera orthoCam;
    perspectiveCamera perspCam;
    bool usePerspective = true;

    scene() {}


    void addGroundPlane() {
        // Ground plane at y = -150, facing upward (normal = 0,1,0)
        plane ground(-0, -150, 0, 0, 1, 0, 200, 200, 200, 32.0f, 0.4f, 0.1f);
        planes.push_back(ground);
    }


    void addObject() {
        // Position spheres so they sit on the ground plane (y = -150)
        // Sphere bottom should touch the plane


        // Magenta sphere (radius 100, so center at y = -50 to sit on ground at y = -150)
        objects.push_back(sphere(100, -100, -50, 500, 255, 0, 255, 150.0f, 0.7f));

        // Cyan sphere (radius 50, so center at y = -100)
        objects.push_back(sphere(50, 150, -100, 300, 0, 255, 255, 32.0f, 0.6f));

        // Yellow sphere (radius 75, so center at y = -75)
        objects.push_back(sphere(75, 0, -75, 800, 255, 255, 0, 16.0f, 0.3f));

        // No boring RGB here, we do CYMK around these parts =)
    }


    void addPerspectiveCamera(float x, float y, float z,
                             float dirX, float dirY, float dirZ,
                             float fov = 60.0f) {
        perspCam = perspectiveCamera(128, 128, x, y, z,
                                    Eigen::Vector3d(dirX, dirY, dirZ), fov);
        usePerspective = true;
    }
    pixel calculateReflection(const point&hitPoint, const Eigen::Vector3f& normal, const ray& incomingRay, void* originalObj, int depth = 0) {
        if (depth > 3) {
            return pixel(128, 168, 255); // default to sky color if too much reflection depth
        }

        // Calculate reflection direction using: R = I - 2(I·N)N
    Eigen::Vector3f incidentDir = incomingRay.dir.normalized();
    Eigen::Vector3f reflectionDir = incidentDir - 2.0f * (incidentDir.dot(normal)) * normal;
    reflectionDir = reflectionDir.normalized();

    // Create reflection ray slightly offset from surface to avoid self-intersection
    float epsilon = 0.1f;
    point reflectionOrigin = point(
        hitPoint.x + normal.x() * epsilon,
        hitPoint.y + normal.y() * epsilon,
        hitPoint.z + normal.z() * epsilon
    );

    ray reflectionRay(reflectionOrigin.x, reflectionOrigin.y, reflectionOrigin.z,
                     reflectionDir.x(), reflectionDir.y(), reflectionDir.z());

    // Trace the reflection ray
    bool hitAny = false;
    float closestDist = 1e10;
    pixel reflectionColor = pixel(128, 168, 255); // Sky blue default

    // So many variables...
    Eigen::Vector3f closestNormal;
    pixel closestBaseColor;
    float closestShininess;
    float closestSpecularStrength;
    pixel closestSpecularColor;
    point closestHitPoint;
    void* closestObject;
    float closestReflectivity = 0.0f;

    // Check sphere intersections
    for (auto& obj : objects) {
        if (&obj == originalObj) continue; // Don't reflect off ourselves

        point p = obj.intersect(reflectionRay);
        if (p != point(0,0,0)) {
            float dist = sqrt(pow(p.x - reflectionRay.e.x, 2) +
                             pow(p.y - reflectionRay.e.y, 2) +
                             pow(p.z - reflectionRay.e.z, 2));

            if (!hitAny || dist < closestDist) {
                hitAny = true;
                closestObject = &obj;
                closestDist = dist;
                closestHitPoint = p;
                closestNormal = obj.getNormal(p);
                closestBaseColor = obj.color;
                closestShininess = obj.shininess;
                closestSpecularStrength = obj.specularStrength;
                closestSpecularColor = obj.specularColor;
                closestReflectivity = 0.0f; // Spheres don't reflect by default
            }
        }
    }

    // Check plane intersections
    for (auto& planeObj : planes) {
        if (&planeObj == originalObj) continue; // Don't reflect off ourselves

        point p = planeObj.intersect(reflectionRay);
        if (p != point(0,0,0)) {
            float dist = sqrt(pow(p.x - reflectionRay.e.x, 2) +
                             pow(p.y - reflectionRay.e.y, 2) +
                             pow(p.z - reflectionRay.e.z, 2));

            if (!hitAny || dist < closestDist) {
                hitAny = true;
                closestObject = &planeObj;
                closestDist = dist;
                closestHitPoint = p;
                closestNormal = planeObj.getNormal(p);
                closestBaseColor = planeObj.getColorAt(p);
                closestShininess = planeObj.shininess;
                closestSpecularStrength = planeObj.specularStrength;
                closestSpecularColor = planeObj.specularColor;
                closestReflectivity = planeObj.getReflectivity();
            }
        }
    }

    // If we hit something, calculate its color
    if (hitAny) {
        // Calculate base lighting
        pixel baseColor = calculateLighting(closestHitPoint, closestNormal,
                                          closestBaseColor, closestShininess,
                                          closestSpecularStrength, closestSpecularColor,
                                          reflectionRay, closestObject);

        // Add reflections if the surface is reflective
        if (closestReflectivity > 0.0f) {
            pixel nestedReflection = calculateReflection(closestHitPoint, closestNormal,
                                                        reflectionRay, closestObject, depth + 1);

            // Blend base color with reflection
            float reflStrength = closestReflectivity;
            reflectionColor = pixel(
                (int)(baseColor.r * (1.0f - reflStrength) + nestedReflection.r * reflStrength),
                (int)(baseColor.g * (1.0f - reflStrength) + nestedReflection.g * reflStrength),
                (int)(baseColor.b * (1.0f - reflStrength) + nestedReflection.b * reflStrength)
            );
        } else {
            reflectionColor = baseColor;
        }
    }

    return reflectionColor;

    }

    // Enhanced lighting calculation that works with both spheres and planes
    pixel calculateLighting(const point& hitPoint, const Eigen::Vector3f& normal,
                           const pixel& baseColor, float shininess,
                           float specularStrength, const pixel& specularColor,
                           const ray& viewRay, void* closestObj) {
        // Calculate view direction
        Eigen::Vector3f viewDir(viewRay.e.x - hitPoint.x,
                               viewRay.e.y - hitPoint.y,
                               viewRay.e.z - hitPoint.z);
        viewDir = viewDir.normalized();

        float totalR = 0, totalG = 0, totalB = 0;
        float ambientStrength = 0.2f;

        // Add ambient lighting
        totalR += baseColor.r * ambientStrength;
        totalG += baseColor.g * ambientStrength;
        totalB += baseColor.b * ambientStrength;

        // Calculate contribution from each light
        for (const auto& lightSource : lights) {
            if (!isInShadow(hitPoint, normal, lightSource, closestObj)) {
                Eigen::Vector3f lightDir = lightSource.getLightDirection(hitPoint);
                float distance = lightSource.getDistance(hitPoint);
                float lightIntensity = lightSource.getIntensityAtDistance(distance);

                // Diffuse lighting
                float diffuse = max(0.0f, normal.dot(lightDir));

                // Specular lighting (Blinn-Phong)
                float specular = 0.0f;
                if (diffuse > 0.0f) {
                    Eigen::Vector3f halfwayDir = (lightDir + viewDir).normalized();
                    float specularIntensity = max(0.0f, normal.dot(halfwayDir));
                    specular = pow(specularIntensity, shininess) * specularStrength;
                }

                // Apply diffuse lighting
                totalR += baseColor.r * diffuse * lightIntensity * (lightSource.color.r / 255.0f);
                totalG += baseColor.g * diffuse * lightIntensity * (lightSource.color.g / 255.0f);
                totalB += baseColor.b * diffuse * lightIntensity * (lightSource.color.b / 255.0f);

                // Apply specular lighting
                totalR += specularColor.r * specular * lightIntensity * (lightSource.color.r / 255.0f);
                totalG += specularColor.g * specular * lightIntensity * (lightSource.color.g / 255.0f);
                totalB += specularColor.b * specular * lightIntensity * (lightSource.color.b / 255.0f);
            }
        }

        // Clamp values
        int finalR = min(255, max(0, (int)totalR));
        int finalG = min(255, max(0, (int)totalG));
        int finalB = min(255, max(0, (int)totalB));

        return pixel(finalR, finalG, finalB);
    }

    // Shadow check
    bool isInShadow(const point& hitPoint, const Eigen::Vector3f& surfaceNormal, const light& lightSource, void* originalObj) {
        Eigen::Vector3f lightDir = lightSource.getLightDirection(hitPoint);
        float lightDistance = lightSource.getDistance(hitPoint);

        float epsilon = 0.1f;
        point shadowRayOrigin = point(
            hitPoint.x + surfaceNormal.x() * epsilon,
            hitPoint.y + surfaceNormal.y() * epsilon,
            hitPoint.z + surfaceNormal.z() * epsilon
        );

        ray shadowRay(shadowRayOrigin.x, shadowRayOrigin.y, shadowRayOrigin.z,
                     lightDir.x(), lightDir.y(), lightDir.z());

        // Check sphere occlusions
        for (const auto& obj : objects) {
            if (&obj == originalObj) {
                continue;  // Don't check shadow against itself!! Learned this the hard way...
            }

            point intersection = obj.intersect(shadowRay);
            if (intersection != point(0,0,0)) {
                float intersectionDistance = sqrt(
                    pow(intersection.x - shadowRayOrigin.x, 2) +
                    pow(intersection.y - shadowRayOrigin.y, 2) +
                    pow(intersection.z - shadowRayOrigin.z, 2)
                );

                if (intersectionDistance < lightDistance - epsilon) {
                    return true;
                }
            }
        }

        // Check plane occlusions
        for (const auto& planeObj : planes) {
            point intersection = planeObj.intersect(shadowRay);
            if (intersection != point(0,0,0)) {
                float intersectionDistance = sqrt(
                    pow(intersection.x - shadowRayOrigin.x, 2) +
                    pow(intersection.y - shadowRayOrigin.y, 2) +
                    pow(intersection.z - shadowRayOrigin.z, 2)
                );

                if (intersectionDistance < lightDistance - epsilon) {
                    return true;
                }
            }
        }

        return false;
    }
    void addLights() {
     // Hardcoded light source, a real program would probably have less hardcoded stuff, but hey...
        lights.push_back(light(100, 100, 300, 255, 255, 255, 20.f));
    }


    image initRender() {
        image result(512, 512);

        if (lights.empty()) {
            addLights();
        }

        if (planes.empty()) {
            addGroundPlane();
        }

        vector<ray>& currentRays = usePerspective ? perspCam.eyeRays : orthoCam.eyeRays;

        for (int i = 0; i < currentRays.size(); i++) {
            bool hitAny = false;
            float closestDist = 1e10;
            pixel finalColor = pixel(128, 168, 255); // Sky blue background

            // Variables to store the closest hit information
            Eigen::Vector3f closestNormal;
            pixel closestBaseColor;
            float closestShininess;
            float closestSpecularStrength;
            pixel closestSpecularColor;
            point closestHitPoint;
            void* closestObject;
            float closestReflectivity = 0.0f;

            // Check sphere intersections
            for (auto& obj : objects) {
                point p = obj.intersect(currentRays[i]);
                if (p != point(0,0,0)) {
                    float dist = sqrt(pow(p.x - currentRays[i].e.x, 2) +
                                     pow(p.y - currentRays[i].e.y, 2) +
                                     pow(p.z - currentRays[i].e.z, 2));

                    if (!hitAny || dist < closestDist) {
                        hitAny = true;
                        closestObject = &obj;
                        closestDist = dist;
                        closestHitPoint = p;
                        closestNormal = obj.getNormal(p);
                        closestBaseColor = obj.color;
                        closestShininess = obj.shininess;
                        closestSpecularStrength = obj.specularStrength;
                        closestSpecularColor = obj.specularColor;
                        closestReflectivity = 0.0f; // Spheres don't reflect by default
                    }
                }
            }

            // Check plane intersections
            for (auto& planeObj : planes) {
                point p = planeObj.intersect(currentRays[i]);
                if (p != point(0,0,0)) {
                    float dist = sqrt(pow(p.x - currentRays[i].e.x, 2) +
                                     pow(p.y - currentRays[i].e.y, 2) +
                                     pow(p.z - currentRays[i].e.z, 2));

                    if (!hitAny || dist < closestDist) {
                        hitAny = true;
                        closestObject = &planeObj;
                        closestDist = dist;
                        closestHitPoint = p;
                        closestNormal = planeObj.getNormal(p);
                        closestBaseColor = planeObj.getColorAt(p); // Use checkerboard pattern
                        closestShininess = planeObj.shininess;
                        closestSpecularStrength = planeObj.specularStrength;
                        closestSpecularColor = planeObj.specularColor;
                        closestReflectivity = planeObj.getReflectivity();
                    }
                }
            }

            // Calculate lighting if we hit something
            if (hitAny) {
                void* hitObj = closestObject;
                finalColor = calculateLighting(closestHitPoint, closestNormal,
                                             closestBaseColor, closestShininess,
                                             closestSpecularStrength, closestSpecularColor,
                                             currentRays[i], hitObj);
                if (closestReflectivity > 0.0f) {
                    pixel reflectionColor = calculateReflection(closestHitPoint, closestNormal, currentRays[i], closestObject);

                    float reflStrength = closestReflectivity;
                    finalColor = pixel(
                        (int)(finalColor.r * (1.0f - reflStrength) + reflectionColor.r * reflStrength),
                        (int)(finalColor.g * (1.0f - reflStrength) + reflectionColor.g * reflStrength),
                        (int)(finalColor.b * (1.0f - reflStrength) + reflectionColor.b * reflStrength));

                }

            }

            // Set pixel color
            int pixelIndex = i * 3;
            result.pixels[pixelIndex] = finalColor.r;
            result.pixels[pixelIndex + 1] = finalColor.g;
            result.pixels[pixelIndex + 2] = finalColor.b;
        }

        return result;
    }
};
