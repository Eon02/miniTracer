// Based on templates from learnopengl.com
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "rayTrace.h"
#include <iostream>

// Add this function near the top after the includes
struct ViewPoint {
    float x, y, z;
    float dirX, dirY, dirZ;
    std::string name;
};

// Define preset viewpoints
std::vector<ViewPoint> getViewPoints() {
    return {
        {0, 0, 200, 1, 0, 1, "default"},
        {0, 0, 200, 1, 0, 1, "ortho"},  // Default forward view
        {0, 600, 0, 0, -1, 0, "back"},           // View from behind
        {300, 300, 0, -1, 0, 0, "right"},          // View from right side
        {-300, 300, 0, 1, 0, 0, "left"},           // View from left side
        {0, 300, 500, 0, -1, 0, "top"},            // View from above
        {0, 300, -300, 0, 0, 1, "bottom"},         // View from below
        {200, -200, 200, -1, 1, -1, "corner"},   // Diagonal corner view
        {0, -500, 0, 0, 1, 0, "far"},            // Far back view
        {50, 100, 100, -0.5, -1, -1, "close"}
        // Close angled view
    };
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [viewpoint]\n";
    std::cout << "Available viewpoints:\n";
    
    auto viewpoints = getViewPoints();
    for (const auto& vp : viewpoints) {
        std::cout << "  " << vp.name << " - Position(" << vp.x << ", " << vp.y << ", " << vp.z << ")\n";
    }
    std::cout << "\nExample: " << programName << " corner\n";
}

ViewPoint parseViewPoint(int argc, char* argv[]) {
    auto viewpoints = getViewPoints();
    
    // default viewpoint
    ViewPoint selectedView = viewpoints[0];
    
    if (argc > 1) {
        std::string requestedView = argv[1];
        
        // help
        if (requestedView == "-h" || requestedView == "--help" || requestedView == "help") {
            printUsage(argv[0]);
            exit(0);
        }
        

        bool found = false;
        for (const auto& vp : viewpoints) {
            if (vp.name == requestedView) {
                selectedView = vp;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cout << "Unknown viewpoint: " << requestedView << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
    }
    // debug, but also just generally helpful
    std::cout << "Using viewpoint: " << selectedView.name 
              << " at position (" << selectedView.x << ", " << selectedView.y << ", " << selectedView.z << ")\n";
    
    return selectedView;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;


const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aColor;\n"
    "layout (location = 2) in vec2 aTexCoord;\n"
    "out vec3 ourColor;\n"
    "out vec2 TexCoord;\n"
    "void main()\n"
    "{\n"
	"gl_Position = vec4(aPos, 1.0);\n"
	"ourColor = aColor;\n"
	"TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
    "}\0";

const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec3 ourColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main()\n"
    "{\n"
    "   FragColor = texture(texture1, TexCoord);\n"
    "}\n\0";

// Camera movement and rotation variables
bool keys[1024] = {false}; // Track key states
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Key callback function for GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }
}

// Processes input and update camera
void processInput(perspectiveCamera& camera, float deltaTime) {

    float velocity = camera.movementSpeed * deltaTime;
    
    // Eigenvector is great
    Eigen::Vector3d position(camera.pos.x, camera.pos.y, camera.pos.z);
    
    // WASD movement
    if (keys[GLFW_KEY_W]) {
        position += camera.forward * velocity;
    }
    if (keys[GLFW_KEY_S]) {
        position -= camera.forward * velocity;
    }
    if (keys[GLFW_KEY_A]) {
        position -= camera.right * velocity;
    }
    if (keys[GLFW_KEY_D]) {
        position += camera.right * velocity;
    }
    
    // Arrow keys for rotation
    float rotationVelocity = camera.rotationSpeed * deltaTime;
    
    if (keys[GLFW_KEY_LEFT]) {
        camera.yaw -= rotationVelocity;
    }
    if (keys[GLFW_KEY_RIGHT]) {
        camera.yaw += rotationVelocity;
    }
    if (keys[GLFW_KEY_UP]) {
        camera.pitch += rotationVelocity;
    }
    if (keys[GLFW_KEY_DOWN]) {
        camera.pitch -= rotationVelocity;
    }
    
    // Constrain pitch to avoid camera flipping
    if (camera.pitch > 89.0f)
        camera.pitch = 89.0f;
    if (camera.pitch < -89.0f)
        camera.pitch = -89.0f;
    
    // Calculate new look direction based on yaw and pitch
    Eigen::Vector3d newLookDir;
    newLookDir[0] = cos(camera.yaw * M_PI / 180.0f) * cos(camera.pitch * M_PI / 180.0f);
    newLookDir[1] = sin(camera.pitch * M_PI / 180.0f);
    newLookDir[2] = sin(camera.yaw * M_PI / 180.0f) * cos(camera.pitch * M_PI / 180.0f);
    newLookDir.normalize();
    
    // Update camera with new position and look direction
    camera.updateCamera(position[0], position[1], position[2], newLookDir);
}

int main(int argc, char* argv[])
{
    ViewPoint viewpoint = parseViewPoint(argc, argv);
    
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Display RGB Array", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // // GLEW: load all OpenGL function pointers
    glewInit();

    // build and compile the shaders
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
        0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
        0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
       -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
       -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
   };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // load and create a texture
    // -------------------------
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create the image (RGB Array) to be displayed
    const int width  = 128; // keep it in powers of 2!
    const int height = 128; // keep it in powers of 2!
    scene s = scene();

    // Use the selected viewpoint instead of hardcoded values
    // s.cam = orthoCamera(512, 512, viewpoint.x, viewpoint.y, viewpoint.z,
    //                    Eigen::Vector3d(viewpoint.dirX, viewpoint.dirY, viewpoint.dirZ));

    s.addPerspectiveCamera(viewpoint.x, viewpoint.y, viewpoint.z, viewpoint.dirX, viewpoint.dirY, viewpoint.dirZ, 60.0f);
    s.addObject();
    s.addGroundPlane();
    s.addLights();

    if (viewpoint.name == "ortho") {
        s.usePerspective = false;
        s.initRender();

    }












    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // Calculate deltatime of current frame
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        if (s.usePerspective) {
            processInput( s.perspCam, deltaTime);

            glfwPollEvents();
        }



        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind Texture
        glBindTexture(GL_TEXTURE_2D, texture);

        // render container
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Update texture with new render
        image result = s.initRender();
        unsigned char* data = &result.pixels[0];

        if (data)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
        }


        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}