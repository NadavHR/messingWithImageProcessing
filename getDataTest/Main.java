import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class Main{
    public static void main(String[] args) {
        try{
            
            DatagramSocket socket = new DatagramSocket(5800, InetAddress.getByName("0.0.0.0"));
            socket.setBroadcast(true);
            System.out.println("Listen on " + socket.getLocalAddress() + " from " + socket.getInetAddress() + " port " + socket.getBroadcast());
            byte[] buf = new byte[24];
            DatagramPacket packet = new DatagramPacket(buf, buf.length);

            while (true) {
                socket.receive(packet);
                float[] locals = new float[]{(ByteBuffer.wrap(packet.getData()).order(ByteOrder.LITTLE_ENDIAN).getFloat()),
                    (ByteBuffer.wrap(packet.getData()).order(ByteOrder.LITTLE_ENDIAN).getFloat(4)),
                    (ByteBuffer.wrap(packet.getData()).order(ByteOrder.LITTLE_ENDIAN).getFloat(8))};
                    System.out.println(String.format("x: %s, y: %s, z: %s", locals[0], locals[1], locals[2]));
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }
}
