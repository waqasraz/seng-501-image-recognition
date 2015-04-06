package org.seng.image_recognition.tools;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;

/**
 * Abstract class for Tool classes using hadoop
 */
public abstract class HadoopTool extends Configured implements Tool {
    public Configuration configuration;

    @Override
    public void setConf(Configuration configuration) {
        this.configuration = configuration;
    }

    @Override
    public Configuration getConf() {
        return this.configuration;
    }
}
