#ifndef _IVR_AMBIENT_HH_
#define _IVR_AMBIENT_HH_

#include <gazebo/common/common.hh>
#define AMBIENT_MAX 0.7
#define AMBIENT_MIN 0.1
#define STEP_SIZE 0.001
namespace gazebo
{
  class IVRAmbient : public VisualPlugin
  {
    public: IVRAmbient();
    public: virtual ~IVRAmbient();
    public: void Load( rendering::VisualPtr _visual, sdf::ElementPtr /*_sdf*/);
    private: void OnUpdate();
    private: event::ConnectionPtr update_connection_;
  };
}
#endif
