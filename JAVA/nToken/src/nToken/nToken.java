package nToken;
import java.nio.charset.StandardCharsets;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


public class nToken {

    static {
        loadModule("libnToken.so.1");
    }


    public  native byte[] callNToken(byte[] text, byte[] searchQuery);

    public   nToken(String[] text, String searchQuery){
        List<byte> all = new ArrayList<byte>();
        for(int i = 0 ; i < text.length ; i++){
            List temp = Arrays.asList(text[i].getBytes(StandardCharsets.UTF_8));
            all.addAll(new ArrayList<>(temp) );
            all.add((byte) 0xF4); //U+10FFFF
            all.add((byte) 0x8F);
            all.add((byte) 0xBF);
            all.add((byte) 0xBF);
        }
        byte[] arr = new byte[all.size()];
        all.toArray(arr);
        callNToken(arr, searchQuery.getBytes(StandardCharsets.UTF_8));
    }

    private static void loadModule(String mod){
        try {
            InputStream in = Thread.currentThread().getContextClassLoader().getResourceAsStream(mod);
            File dll = File.createTempFile(mod, ".mod");
            FileOutputStream out = new FileOutputStream(dll);
            int i;
            byte [] buf = new byte[1024];
            while((i=in.read(buf))!=-1) {
                out.write(buf,0,i);
            }
            in.close();
            out.close();
            System.load(dll.toString());
            dll.deleteOnExit();
        }catch (Exception e) {
            System.err.println("load module "+mod+" error!");
            e.printStackTrace();
        }
    }





    public static void main(String[] args){
        nToken token = new nToken("1234", "5678");
    }
}
