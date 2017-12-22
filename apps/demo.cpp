#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/cuda/marching_cubes.hpp>
#include <io/capture.hpp>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

using namespace kfusion;
using namespace std;

struct KinFuApp
{
    void print_help()
    {
        cout << endl;
        cout << "kinfu_remake app hotkeys" << endl;
        cout << "=================" << endl;
        cout << "    H    : print VTK help" << endl;
        cout << "    K    : print this help" << endl;
        cout << "    Q    : exit" << endl;
        cout << "    T    : take cloud and save as a .pcd file" << endl;
        cout << "    M    : take mesh and save as a .ply file" << endl;
        cout << "    I    : toggle iteractive mode" << endl;
        cout << "=================" << endl;
        cout << endl;
    } 

    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_and_save_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;

        if(event.code == 'm' || event.code == 'M')
            kinfu.take_mesh(*kinfu.kinfu_);

        if(event.code == 's' || event.code == 'S')
            kinfu.write_mesh(*kinfu.kinfu_);

        if(event.code == 'k' || event.code == 'K')
            kinfu.print_help();
    }

    KinFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), capture_ (source), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    KinFuApp(OpenNISource& source, const KinFuParams& params) : exit_ (false),  iteractive_mode_(false), capture_ (source), pause_(false)
    {
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 3;
        if (iteractive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void take_and_save_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer_);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        if (kinfu.params().integrate_color) {
            kinfu.color_volume()->fetchColors(cloud, color_buffer_);
            cv::Mat color_host(1, (int)cloud.size(), CV_8UC4);
            cloud.download(cloud_host.ptr<Point>());
            color_buffer_.download(color_host.ptr<RGB>());
            viz.showWidget("cloud", cv::viz::WCloud(cloud_host, color_host));
        } else
        {
            cloud.download(cloud_host.ptr<Point>());
            viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        }
        save_next_cloud(cloud_host);
    }


    void save_next_cloud(const cv::Mat &cloud_host)
    {
        const std::string dirname = "pointclouds";
        const std::string filenames_template = "cloud";

        // Pointclouds dir
        if (!fs::exists(dirname))
            fs::create_directory(dirname);

        // Get filenames
        std::vector<std::string> filenames;
        for (fs::directory_iterator it(dirname); it!=fs::directory_iterator(); ++it)
            if (is_regular_file(it->path())) filenames.push_back(it->path().filename().string());

        // Determine filename
        std::string filename;
        if (filenames.empty()) filename = filenames_template + "_0.pcd";
        else
        {
            int max = -1;
            for (auto flnm : filenames)
            {
                size_t pos = flnm.find(filenames_template);
                if (pos != std::string::npos)
                {
                    size_t flood = flnm.find("_");
                    size_t dot = flnm.find(".");
                    int curr = std::stoi(flnm.substr(flood + 1, dot-flood - 1));
                    if (max < curr) max = curr;
                }
            }
            filename = filenames_template + "_" + std::to_string(max+1) + ".pcd";
        }
        fs::path filepath = fs::path(dirname) / fs::path(filename);

        // Save pcd file
        ofstream myfile;
        myfile.open(filepath.string());
        myfile << "# .PCD v.5 - Point Cloud Data file format\n";
        myfile << "FIELDS x y z\n";
        myfile << "SIZE 4 4 4\n";
        myfile << "TYPE F F F\n";
        myfile << "WIDTH " << cloud_host.cols << "\n";
        myfile << "HEIGHT 1\n";
        myfile << "POINTS " << cloud_host.cols << "\n";
        myfile << "DATA ascii\n";
        for (int i=0; i<cloud_host.cols; ++i)
            myfile << cloud_host.at<Point>(i).x << " " << cloud_host.at<Point>(i).y << " " <<  cloud_host.at<Point>(i).z << "\n";
        myfile.close();

        // Info
        cout << "Succesfully saved : " << filename << " with " << cloud_host.cols << " points" << endl;
    }

    void write_mesh(KinFu& kinfu )
    {

        string outFilename = "mesh.ply";
        //first take mesh

        if (!marching_cubes_)
            marching_cubes_ = cv::Ptr<cuda::MarchingCubes>(new cuda::MarchingCubes());

        cuda::DeviceArray<Point> triangles = marching_cubes_->run(kinfu.tsdf(), triangles_buffer_);
        int n_vert = triangles.size();

        cv::viz::Mesh mesh;
        mesh.cloud.create(1, n_vert, CV_32FC4);
        mesh.polygons.create(1, 4*n_vert/3, CV_32SC1);

        for (int i = 0; i < n_vert/3; ++i) {
            mesh.polygons.at<int>(4*i) = 3;
            mesh.polygons.at<int>(4*i+1) = 3*i;
            mesh.polygons.at<int>(4*i+2) = 3*i+1;
            mesh.polygons.at<int>(4*i+3) = 3*i+2;
        }

        cv::Mat mesh_colors(1, n_vert, CV_8UC4);

        if (kinfu.params().integrate_color)
        {
            kinfu.color_volume()->fetchColors(triangles, color_buffer_);
            color_buffer_.download(mesh_colors.ptr<RGB>());
            mesh.colors = mesh_colors;
        }

        triangles.download(mesh.cloud.ptr<Point>());

        ofstream outFile( outFilename.c_str() );

        if ( !outFile )
        {
            cerr << "Error opening output file: " << outFilename << "!" << endl;
            exit( 1 );
        }

        ////
        // Header
        ////
        const int pointNum    =  mesh.cloud.cols;
        const int triangleNum =  mesh.polygons.cols/4; //polygons is a Mat where each column has only 1 value
        cout<<pointNum<<" "<<triangleNum<<endl;
        
        outFile << "ply" << endl;
        outFile << "format ascii 1.0" << endl;
        outFile << "element vertex " << pointNum << endl;
        outFile << "property float x" << endl;
        outFile << "property float y" << endl;
        outFile << "property float z" << endl;
        outFile << "property uchar red" << endl;
        outFile << "property uchar green" << endl;
        outFile << "property uchar blue" << endl;
        outFile << "element face " << triangleNum << endl;
        outFile << "property list uchar int vertex_index" << endl;
        outFile << "end_header" << endl;

        ////
        // Points and colors
        ////
        vector<cv::Mat> channels_cloud(4);
        vector<cv::Mat> channels_colors(4);
        split(mesh.cloud,channels_cloud);
        split(mesh.colors,channels_colors);
        for ( int i = 0; i < pointNum; i++ )
        {
            outFile << channels_cloud[0].at<float>(0,i)<<" ";       //x
            outFile << channels_cloud[1].at<float>(0,i)<<" ";       //y
            outFile << channels_cloud[2].at<float>(0,i)<<" ";       //z
            outFile << (int)channels_colors[2].at<uchar>(0,i)<<" "; //b
            outFile << (int)channels_colors[1].at<uchar>(0,i)<<" "; //g
            outFile << (int)channels_colors[0].at<uchar>(0,i)<<" "; //r

            outFile << endl;
        }

        ////
        // Triangles
        ////
        for ( int i = 0; i < triangleNum*4; i+=4 )
        {
            outFile << mesh.polygons.at<int>(0,i+0)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+1)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+2)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+3)<<" ";             

            outFile << endl;
        }
    }

    void take_mesh(KinFu& kinfu)
    {
        if (!marching_cubes_)
            marching_cubes_ = cv::Ptr<cuda::MarchingCubes>(new cuda::MarchingCubes());

        cuda::DeviceArray<Point> triangles = marching_cubes_->run(kinfu.tsdf(), triangles_buffer_);
        int n_vert = triangles.size();

        cv::viz::Mesh mesh;
        mesh.cloud.create(1, n_vert, CV_32FC4);
        mesh.polygons.create(1, 4*n_vert/3, CV_32SC1);

        for (int i = 0; i < n_vert/3; ++i) {
            mesh.polygons.at<int>(4*i) = 3;
            mesh.polygons.at<int>(4*i+1) = 3*i;
            mesh.polygons.at<int>(4*i+2) = 3*i+1;
            mesh.polygons.at<int>(4*i+3) = 3*i+2;
        }

        cv::Mat mesh_colors(1, n_vert, CV_8UC4);

        if (kinfu.params().integrate_color)
        {
            kinfu.color_volume()->fetchColors(triangles, color_buffer_);
            color_buffer_.download(mesh_colors.ptr<RGB>());
            mesh.colors = mesh_colors;
        }

        triangles.download(mesh.cloud.ptr<Point>());

        

        viz.showWidget("cloud", cv::viz::WMesh(mesh));
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
            bool has_frame = capture_.grab(depth, image);
            if (!has_frame)
                return std::cout << "Can't grab" << std::endl, false;

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            color_device_.upload(image.data, image.step, image.rows, image.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                if (kinfu.params().integrate_color)
                    has_image = kinfu(depth_device_, color_device_);
                else
                    has_image = kinfu(depth_device_);
            }

            if (has_image)
                show_raycasted(kinfu);

            show_depth(depth);
            if (kinfu.params().integrate_color)
                cv::imshow("Image", image);

            if (!iteractive_mode_)
                viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);
            
            switch(key)
            {
            case 't': case 'T' : take_and_save_cloud(kinfu); break;
            case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
            case 'm': case 'M' : take_mesh(kinfu); break;
            case 27: exit_ = true; break;
            case 32: pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::Image color_device_;
    cuda::DeviceArray<Point> cloud_buffer_;
    cuda::DeviceArray<RGB> color_buffer_;
    cv::Ptr<cuda::MarchingCubes> marching_cubes_;
    cuda::DeviceArray<Point> triangles_buffer_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    OpenNISource capture;
    capture.open (0);
   
    KinFuParams custom_params = KinFuParams::default_params();
    custom_params.integrate_color = true;

    KinFuApp app (capture, custom_params);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
