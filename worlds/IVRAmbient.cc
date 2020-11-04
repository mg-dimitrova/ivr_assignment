#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/rendering/rendering.hh>
#include "IVRAmbient.hh"

using namespace gazebo;

// Register this plugin with the simulator
GZ_REGISTER_VISUAL_PLUGIN(IVRAmbient)


IVRAmbient::IVRAmbient():VisualPlugin()
{
}
IVRAmbient::~IVRAmbient()
{
}


void IVRAmbient::Load( rendering::VisualPtr _visual, sdf::ElementPtr /*_sdf*/)
{
      this->update_connection_ = event::Events::ConnectRender(
            boost::bind(&IVRAmbient::OnUpdate, this));
}
void IVRAmbient::OnUpdate()
{
      static double amb=0.1;
      static int direction=1;
      rendering::ScenePtr scene = rendering::get_scene();
      if (amb>AMBIENT_MAX) direction=-1;
      if (amb<AMBIENT_MIN) direction = 1; 
      amb+= direction*STEP_SIZE;
      scene->SetAmbientColor(common::Color(amb,amb,amb,1));
}
